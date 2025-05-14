// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title OnChainSNN
 * @dev A fully on-chain implementation of a Spiking Neural Network
 */
contract OnChainSNN is Ownable {
    // Fixed-point arithmetic precision (18 decimals)
    int256 constant FP_SCALING = 10**18;

    // Network parameters
    uint256 constant MAX_NEURONS = 128;  // Maximum neurons in a network
    uint256 constant MAX_SYNAPSES_PER_NEURON = 32;  // Maximum connections per neuron
    int256 constant MAX_WEIGHT = 1000 * FP_SCALING;  // Maximum weight value (positive or negative)
    int256 constant MIN_WEIGHT = -1000 * FP_SCALING;  // Minimum weight value

    // Neuron types
    enum NeuronType { 
        INPUT,      // Input neuron (receives external input)
        EXCITATORY, // Excitatory neuron (increases potential of connected neurons)
        INHIBITORY  // Inhibitory neuron (decreases potential of connected neurons)
    }

    // Neuron structure - optimized for gas efficiency with careful use of types
    struct Neuron {
        bool active;                   // Whether neuron is active in network
        NeuronType neuronType;         // Type of neuron
        int256 membranePotential;      // Current membrane potential (fixed-point)
        int256 restingPotential;       // Resting potential (fixed-point)
        int256 threshold;              // Firing threshold (fixed-point)
        uint8 refractoryPeriod;        // Refractory period in time steps
        uint8 refractoryCounter;       // Current refractory counter
        bool hasFired;                 // Whether neuron has fired in current time step
        uint8 layer;                   // Neuron's layer in the network
    }

    // Synapse structure
    struct Synapse {
        uint16 targetNeuronId;         // Target neuron ID
        int256 weight;                 // Synaptic weight (fixed-point)
        int256 lastWeightUpdate;       // When weight was last updated (for STDP)
    }

    // SNN Network structure
    struct SNNetwork {
        string name;                   // Network name
        uint16 neuronCount;            // Total neurons in network
        uint8 inputCount;              // Number of input neurons
        uint8 outputCount;             // Number of output neurons
        uint8 layerCount;              // Number of layers
        uint32 timeStep;               // Current simulation time step
        bool learningEnabled;          // Whether STDP learning is enabled
    }

    // Spike record structure
    struct Spike {
        uint16 neuronId;               // Neuron that spiked
        uint32 timeStep;               // Time step when spike occurred
    }

    // Storage
    mapping(uint256 => SNNetwork) public networks;       // Network ID -> Network
    mapping(uint256 => mapping(uint16 => Neuron)) public neurons;  // Network ID -> Neuron ID -> Neuron
    mapping(uint256 => mapping(uint16 => Synapse[])) public synapses; // Network ID -> Neuron ID -> Outgoing Synapses
    mapping(uint256 => Spike[]) public spikeHistory;     // Network ID -> Spike history (recent spikes)
    mapping(uint256 => mapping(uint16 => uint32[])) public neuronSpikeHistory; // Network ID -> Neuron ID -> Spike times

    uint256 private nextNetworkId = 1;

    // Events
    event NetworkCreated(uint256 indexed networkId, string name);
    event NeuronAdded(uint256 indexed networkId, uint16 neuronId, NeuronType neuronType, uint8 layer);
    event SynapseCreated(uint256 indexed networkId, uint16 sourceNeuronId, uint16 targetNeuronId, int256 weight);
    event NetworkSimulated(uint256 indexed networkId, uint32 timeStep, uint16 spikeCount);
    event NeuronFired(uint256 indexed networkId, uint16 neuronId, uint32 timeStep);
    event WeightUpdated(uint256 indexed networkId, uint16 sourceNeuronId, uint16 targetNeuronId, int256 newWeight);

    /**
     * @dev Create a new SNN network
     * @param name Network name
     * @param inputCount Number of input neurons
     * @param outputCount Number of output neurons
     * @return networkId ID of the created network
     */
    function createNetwork(
        string memory name,
        uint8 inputCount,
        uint8 outputCount
    ) public returns (uint256) {
        require(inputCount > 0, "Must have at least one input");
        require(outputCount > 0, "Must have at least one output");
        require(inputCount + outputCount <= MAX_NEURONS, "Too many neurons");

        uint256 networkId = nextNetworkId++;

        networks[networkId] = SNNetwork({
            name: name,
            neuronCount: 0,
            inputCount: inputCount,
            outputCount: outputCount,
            layerCount: 2,  // Default: input and output layers
            timeStep: 0,
            learningEnabled: false  // Learning disabled by default
        });

        // Create input neurons
        for (uint16 i = 0; i < inputCount; i++) {
            _createNeuron(
                networkId,
                NeuronType.INPUT,
                0,  // Layer 0 = input layer
                0,  // Input neurons have no refractory period
                0,  // Resting potential
                FP_SCALING  // Threshold = 1.0
            );
        }

        // Create output neurons
        for (uint16 i = 0; i < outputCount; i++) {
            _createNeuron(
                networkId,
                NeuronType.EXCITATORY,
                1,  // Layer 1 = output layer
                2,  // Default refractory period
                0,  // Resting potential
                FP_SCALING  // Threshold = 1.0
            );
        }

        emit NetworkCreated(networkId, name);
        return networkId;
    }

    /**
     * @dev Add a hidden layer to the network
     * @param networkId Network ID
     * @param neuronCount Number of neurons in the layer
     * @param layerPosition Position of the layer (0 = input, layerCount-1 = output)
     */
    function addHiddenLayer(
        uint256 networkId,
        uint8 neuronCount,
        uint8 layerPosition
    ) public {
        SNNetwork storage network = networks[networkId];
        require(neuronCount > 0, "Must add at least one neuron");
        require(network.neuronCount + neuronCount <= MAX_NEURONS, "Too many neurons");
        require(layerPosition > 0 && layerPosition < network.layerCount, "Invalid layer position");

        // Increment layer position for all neurons in higher layers
        for (uint16 i = 0; i < network.neuronCount; i++) {
            if (neurons[networkId][i].layer >= layerPosition) {
                neurons[networkId][i].layer += 1;
            }
        }

        // Create neurons in the new layer
        for (uint16 i = 0; i < neuronCount; i++) {
            // Randomly create either excitatory or inhibitory neurons (75%/25% split)
            NeuronType nType = _pseudoRandom(network.timeStep + i) % 4 > 0 ? 
                NeuronType.EXCITATORY : NeuronType.INHIBITORY;

            _createNeuron(
                networkId,
                nType,
                layerPosition,
                3,  // Default refractory period
                0,  // Resting potential
                FP_SCALING  // Threshold = 1.0
            );
        }

        network.layerCount += 1;
    }

    /**
     * @dev Create a synapse between two neurons
     * @param networkId Network ID
     * @param sourceNeuronId Source neuron ID
     * @param targetNeuronId Target neuron ID
     * @param weight Initial weight value
     */
    function createSynapse(
        uint256 networkId,
        uint16 sourceNeuronId,
        uint16 targetNeuronId,
        int256 weight
    ) public {
        SNNetwork storage network = networks[networkId];
        require(sourceNeuronId < network.neuronCount, "Invalid source neuron");
        require(targetNeuronId < network.neuronCount, "Invalid target neuron");
        require(weight >= MIN_WEIGHT && weight <= MAX_WEIGHT, "Weight out of range");
        require(neurons[networkId][sourceNeuronId].layer < neurons[networkId][targetNeuronId].layer, 
                "Can only connect to neurons in higher layers");
        require(synapses[networkId][sourceNeuronId].length < MAX_SYNAPSES_PER_NEURON, 
                "Too many synapses for source neuron");

        // Adjust weight sign based on neuron type
        if (neurons[networkId][sourceNeuronId].neuronType == NeuronType.INHIBITORY && weight > 0) {
            weight = -weight;  // Inhibitory neurons have negative weights
        }

        // Create the synapse
        synapses[networkId][sourceNeuronId].push(Synapse({
            targetNeuronId: targetNeuronId,
            weight: weight,
            lastWeightUpdate: 0
        }));

        emit SynapseCreated(networkId, sourceNeuronId, targetNeuronId, weight);
    }

    /**
     * @dev Set input values for the network's input layer
     * @param networkId Network ID
     * @param inputs Array of input values
     */
    function setNetworkInputs(
        uint256 networkId,
        int256[] memory inputs
    ) public {
        SNNetwork storage network = networks[networkId];
        require(inputs.length == network.inputCount, "Input count mismatch");

        // Set membrane potential for input neurons
        for (uint16 i = 0; i < network.inputCount; i++) {
            neurons[networkId][i].membranePotential = inputs[i];

            // If input exceeds threshold, neuron will fire in next time step
            neurons[networkId][i].hasFired = inputs[i] >= neurons[networkId][i].threshold;
        }
    }

    /**
     * @dev Run one time step of the network simulation
     * @param networkId Network ID
     */
    function simulateTimeStep(uint256 networkId) public {
        SNNetwork storage network = networks[networkId];
        uint16 spikeCount = 0;

        // Process neurons that have fired in this time step
        for (uint16 i = 0; i < network.neuronCount; i++) {
            Neuron storage neuron = neurons[networkId][i];

            if (neuron.hasFired) {
                // Record the spike
                spikeHistory[networkId].push(Spike({
                    neuronId: i,
                    timeStep: network.timeStep
                }));

                neuronSpikeHistory[networkId][i].push(network.timeStep);
                spikeCount++;

                // Emit spike event
                emit NeuronFired(networkId, i, network.timeStep);

                // Reset membrane potential and enter refractory period
                neuron.membranePotential = neuron.restingPotential;
                neuron.refractoryCounter = neuron.refractoryPeriod;

                // Process outgoing connections
                for (uint j = 0; j < synapses[networkId][i].length; j++) {
                    Synapse storage syn = synapses[networkId][i][j];

                    // Update target neuron's potential
                    Neuron storage targetNeuron = neurons[networkId][syn.targetNeuronId];
                    if (targetNeuron.refractoryCounter == 0) {
                        targetNeuron.membranePotential += syn.weight;
                    }
                }

                // Reset fired flag
                neuron.hasFired = false;
            }
        }

        // Update all neurons for next time step
        for (uint16 i = 0; i < network.neuronCount; i++) {
            Neuron storage neuron = neurons[networkId][i];

            // Decrease refractory counter if neuron is in refractory period
            if (neuron.refractoryCounter > 0) {
                neuron.refractoryCounter--;
            }
            // Check if neuron should fire in next time step
            else if (neuron.membranePotential >= neuron.threshold) {
                neuron.hasFired = true;
            }
            // Leaky integration - decay membrane potential
            else {
                neuron.membranePotential = neuron.membranePotential * (FP_SCALING - FP_SCALING/10) / FP_SCALING;
            }
        }

        // Apply STDP learning if enabled
        if (network.learningEnabled) {
            _applySTDPLearning(networkId);
        }

        network.timeStep++;
        emit NetworkSimulated(networkId, network.timeStep, spikeCount);
    }

    /**
     * @dev Get the output values of the network
     * @param networkId Network ID
     * @return outputValues Array of output values
     */
    function getNetworkOutputs(uint256 networkId) public view returns (int256[] memory) {
        SNNetwork storage network = networks[networkId];
        int256[] memory outputs = new int256[](network.outputCount);

        // Get membrane potentials of output neurons
        for (uint16 i = 0; i < network.outputCount; i++) {
            // Output neurons are after input neurons
            uint16 neuronId = network.inputCount + i;
            outputs[i] = neurons[networkId][neuronId].membranePotential;
        }

        return outputs;
    }

    /**
     * @dev Get recent spike history for the network
     * @param networkId Network ID
     * @param maxSpikes Maximum number of spikes to return
     * @return recentSpikes Array of recent spikes
     */
    function getRecentSpikes(
        uint256 networkId,
        uint16 maxSpikes
    ) public view returns (Spike[] memory) {
        Spike[] storage spikes = spikeHistory[networkId];

        // Calculate how many spikes to return
        uint256 spikeCount = spikes.length;
        uint256 resultCount = spikeCount < maxSpikes ? spikeCount : maxSpikes;

        // Create result array
        Spike[] memory result = new Spike[](resultCount);

        // Copy most recent spikes to result array
        for (uint256 i = 0; i < resultCount; i++) {
            result[i] = spikes[spikeCount - resultCount + i];
        }

        return result;
    }

    /**
     * @dev Enable or disable STDP learning for the network
     * @param networkId Network ID
     * @param enabled Whether learning should be enabled
     */
    function setLearningEnabled(
        uint256 networkId,
        bool enabled
    ) public {
        networks[networkId].learningEnabled = enabled;
    }

    /**
     * @dev Generate a text encoding of the network structure
     * @param networkId Network ID
     * @return encoding Text encoding of network structure
     */
    function getNetworkEncoding(uint256 networkId) public view returns (string memory) {
        SNNetwork storage network = networks[networkId];

        // Build a simple encoding of the network structure
        string memory encoding = string(abi.encodePacked(
            "Network: ", network.name,
            ", Neurons: ", _toString(network.neuronCount),
            ", Layers: ", _toString(network.layerCount),
            ", Time: ", _toString(network.timeStep)
        ));

        return encoding;
    }

    /**
     * @dev Apply Spike-Timing-Dependent Plasticity learning
     * @param networkId Network ID
     */
    function _applySTDPLearning(uint256 networkId) internal {
        SNNetwork storage network = networks[networkId];

        // Need at least 2 spikes to apply STDP
        if (spikeHistory[networkId].length < 2) return;

        // Get recent spikes (last 50 max)
        uint256 spikeCount = spikeHistory[networkId].length;
        uint256 recentCount = spikeCount < 50 ? spikeCount : 50;

        // For each recent spike, find related spikes and update weights
        for (uint256 i = 0; i < recentCount; i++) {
            Spike storage spike = spikeHistory[networkId][spikeCount - recentCount + i];

            // For each synapse from the spiked neuron
            for (uint j = 0; j < synapses[networkId][spike.neuronId].length; j++) {
                Synapse storage syn = synapses[networkId][spike.neuronId][j];

                // Find recent spikes from the target neuron
                uint32[] storage targetSpikes = neuronSpikeHistory[networkId][syn.targetNeuronId];
                if (targetSpikes.length == 0) continue;

                uint256 targetSpikeCount = targetSpikes.length;
                uint32 lastTargetSpike = targetSpikes[targetSpikeCount - 1];

                // Calculate time difference
                int256 timeDiff;
                if (lastTargetSpike > spike.timeStep) {
                    // Post-synaptic neuron fired after pre-synaptic neuron (potentiation)
                    timeDiff = int256(uint256(lastTargetSpike - spike.timeStep));

                    // Apply weight update (simplified STDP)
                    int256 weightDelta = _calculateWeightDelta(timeDiff, true);
                    syn.weight += weightDelta;

                    // Ensure weight stays within bounds
                    if (syn.weight > MAX_WEIGHT) syn.weight = MAX_WEIGHT;

                    emit WeightUpdated(networkId, spike.neuronId, syn.targetNeuronId, syn.weight);
                } else if (lastTargetSpike < spike.timeStep) {
                    // Post-synaptic neuron fired before pre-synaptic neuron (depression)
                    timeDiff = int256(uint256(spike.timeStep - lastTargetSpike));

                    // Apply weight update (simplified STDP)
                    int256 weightDelta = _calculateWeightDelta(timeDiff, false);
                    syn.weight += weightDelta;

                    // Ensure weight stays within bounds
                    if (syn.weight < MIN_WEIGHT) syn.weight = MIN_WEIGHT;

                    emit WeightUpdated(networkId, spike.neuronId, syn.targetNeuronId, syn.weight);
                }

            syn.lastWeightUpdate = int256(uint256(network.timeStep));
            }
        }
    }

    /**
     * @dev Calculate weight delta for STDP
     * @param timeDiff Time difference between spikes
     * @param isLTP Whether this is Long-Term Potentiation (true) or Depression (false)
     * @return delta Weight change
     */
    function _calculateWeightDelta(int256 timeDiff, bool isLTP) internal pure returns (int256) {
        // Simple STDP rule - weight change decreases exponentially with time difference
        int256 maxChange = isLTP ? FP_SCALING / 10 : -(FP_SCALING / 20); // LTP stronger than LTD

        // Calculate exponential decay based on time difference
        // With 5 time steps, weight change reduces to about 37% of max
        int256 timeConstant = 5 * FP_SCALING;
        int256 expDecay = _exponentialDecay(timeDiff * FP_SCALING, timeConstant);

        return (maxChange * expDecay) / FP_SCALING;
    }

    /**
     * @dev Exponential decay function for fixed-point arithmetic
     * @param x Input value
     * @param tau Time constant
     * @return exp(-x/tau) approximation
     */
    function _exponentialDecay(int256 x, int256 tau) internal pure returns (int256) {
        if (x <= 0) return int256(FP_SCALING);
        if (x >= 100 * FP_SCALING) return 0;

        // Calculate exp(-x/tau) using Taylor series approximation
        int256 xDivTau = (x * FP_SCALING) / tau;

        int256 result = FP_SCALING;
        int256 term = FP_SCALING;

        // First 6 terms of Taylor series for e^(-x)
        for (uint i = 1; i <= 6; i++) {
            term = (term * xDivTau) / (int256(i) * FP_SCALING);
            if (i % 2 == 1) {
                result -= term;
            } else {
                result += term;
            }
        }

    return result > 0 ? result : int256(0);
    }

    /**
     * @dev Create a new neuron in the network
     */
    function _createNeuron(
        uint256 networkId,
        NeuronType neuronType,
        uint8 layer,
        uint8 refractoryPeriod,
        int256 restingPotential,
        int256 threshold
    ) internal returns (uint16) {
        SNNetwork storage network = networks[networkId];
        uint16 neuronId = network.neuronCount++;

        neurons[networkId][neuronId] = Neuron({
            active: true,
            neuronType: neuronType,
            membranePotential: restingPotential,
            restingPotential: restingPotential,
            threshold: threshold,
            refractoryPeriod: refractoryPeriod,
            refractoryCounter: 0,
            hasFired: false,
            layer: layer
        });

        emit NeuronAdded(networkId, neuronId, neuronType, layer);
        return neuronId;
    }

    /**
     * @dev Generate a pseudorandom number
     */
    function _pseudoRandom(uint256 seed) internal view returns (uint256) {
        return uint256(keccak256(abi.encodePacked(blockhash(block.number - 1), seed)));
    }

    /**
     * @dev Convert uint to string
     */
    function _toString(uint256 value) internal pure returns (string memory) {
        if (value == 0) {
            return "0";
        }

        uint256 temp = value;
        uint256 digits;

        while (temp != 0) {
            digits++;
            temp /= 10;
        }

        bytes memory buffer = new bytes(digits);
        while (value != 0) {
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }

        return string(buffer);
    }
}
