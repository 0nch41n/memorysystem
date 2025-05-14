// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/utils/Strings.sol";
import "@openzeppelin/contracts/utils/Base64.sol";
import "./OnChainSNN.sol";

/**
 * @title NeuralSOCIAL
 * @dev SOCIAL contract with integrated on-chain Spiking Neural Network
 */
contract NeuralSOCIAL is ERC721, Ownable {
    using Counters for Counters.Counter;
    using Strings for uint256;
    Counters.Counter private _tokenIdCounter;

    // Reference to the SNN contract
    OnChainSNN public snnContract;

    // Neural encoding for memories
    struct NeuralEncoding {
        uint256 networkId;              // ID of the SNN network
        uint16[] activatedNeurons;      // Neurons that were activated by this memory
        int256[] conceptActivations;    // Activation levels of different concepts
        uint256 lastProcessed;          // When memory was last processed through SNN
        uint256 neuroplasticityScore;   // Score indicating memory's neuroplasticity
        bytes32 memoryFingerprint;      // Neural fingerprint of the memory
    }

    // Concept registry for neural processing
    struct Concept {
        string name;                    // Concept name
        uint16[] associatedNeurons;     // Neurons associated with this concept
        uint256 activationThreshold;    // Threshold for concept activation
        uint256 lastActivated;          // When concept was last activated
    }

    // Memory Metadata (extended version)
    struct MemoryMetadata {
        string memoryText;              // The actual text of the memory/quote
        string username;                // Username of the creator
        address userWallet;             // Wallet address of the creator
        uint256 qualityScore;           // AI-evaluated quality score (0-100)
        uint256 rewardAmount;           // Amount of NGMI rewarded
        uint256 timestamp;              // When the memory was created
        string[] tags;                  // Tags associated with the memory
        string xPostUrl;                // URL to X (Twitter) post
        NeuralEncoding neuralEncoding;  // Neural representation of memory
    }

    // Spiking Neural Network state
    uint256 public globalSNNId;         // ID of the global SNN for all memories
    uint256 public conceptCount;        // Number of registered concepts

    // Mapping from token ID to its metadata
    mapping(uint256 => MemoryMetadata) public memoryMetadata;

    // Mapping for registered concepts
    mapping(uint256 => Concept) public concepts;

    // Memory networks
    mapping(uint256 => uint256) public memoryNetworks;

    // Mapping from neural fingerprint to token ID
    mapping(bytes32 => uint256) public neuralFingerprints;

    // Authorized minters
    mapping(address => bool) public authorizedMinters;

    // Toggle for on-chain metadata
    bool public useOnChainMetadata = true;

    // Base URI for external metadata
    string private _baseURIExtended = "";

    // Events
    event MemoryMinted(uint256 indexed tokenId, address indexed creator, string username, uint256 timestamp);
    event MemoryProcessed(uint256 indexed tokenId, uint256 networkId, uint16 activatedNeuronCount);
    event ConceptActivated(uint256 indexed conceptId, uint256 indexed tokenId, uint256 activationLevel);
    event NeuralNetworkInitialized(uint256 networkId, uint16 inputCount, uint16 hiddenCount, uint16 outputCount);
    event MinterAuthorized(address indexed minter, bool status);

    /**
     * @dev Constructor sets the name and symbol of the NFT collection and initializes SNN
     * @param snnAddress Address of the OnChainSNN contract
     */
    constructor(address snnAddress) ERC721("NeuralSOCIAL", "NSOCIAL") {
        require(snnAddress != address(0), "Invalid SNN contract address");
        snnContract = OnChainSNN(snnAddress);

        // Set contract deployer as first authorized minter
        authorizedMinters[msg.sender] = true;
        emit MinterAuthorized(msg.sender, true);

        // Initialize global neural network
        _initializeGlobalNetwork();

        // Initialize basic concepts
        _initializeBasicConcepts();
    }

    /**
     * @dev Modifier to restrict function calls to authorized minters
     */
    modifier onlyAuthorizedMinter() {
        require(authorizedMinters[msg.sender], "Caller is not an authorized minter");
        _;
    }

    /**
     * @dev Add or remove an authorized minter
     * @param minter Address to authorize or deauthorize
     * @param status True to authorize, false to deauthorize
     */
    function setMinterAuthorization(address minter, bool status) public onlyOwner {
        authorizedMinters[minter] = status;
        emit MinterAuthorized(minter, status);
    }

    /**
     * @dev Initialize the global neural network
     */
    function _initializeGlobalNetwork() internal {
        // Create a network with 16 input neurons, 32 hidden neurons, and 8 output neurons
        globalSNNId = snnContract.createNetwork("GlobalMemoryNetwork", 16, 8);

        // Add hidden layer
        snnContract.addHiddenLayer(globalSNNId, 32, 1);

        // Connect input to hidden layer with random weights
        for (uint16 i = 0; i < 16; i++) {
            for (uint16 j = 0; j < 32; j++) {
                // Random weight between -1.0 and 1.0
                int256 weight = int256(uint256(keccak256(abi.encodePacked(i, j, block.timestamp)))) % (2 * 10**18) - 10**18;
                snnContract.createSynapse(globalSNNId, i, 16 + j, weight);
            }
        }

        // Connect hidden to output layer with random weights
        for (uint16 i = 0; i < 32; i++) {
            for (uint16 j = 0; j < 8; j++) {
                // Random weight between -1.0 and 1.0
                int256 weight = int256(uint256(keccak256(abi.encodePacked(i, j, block.timestamp + 1)))) % (2 * 10**18) - 10**18;
                snnContract.createSynapse(globalSNNId, 16 + i, 16 + 32 + j, weight);
            }
        }

        // Enable learning
        snnContract.setLearningEnabled(globalSNNId, true);

        emit NeuralNetworkInitialized(globalSNNId, 16, 32, 8);
    }

    /**
     * @dev Initialize basic concepts for neural processing
     */
    function _initializeBasicConcepts() internal {
        // Define some basic concepts with associated neurons
        _registerConcept("positive", _generateRandomNeurons(5, 0), 7 * 10**17);  // Threshold 0.7
        _registerConcept("negative", _generateRandomNeurons(5, 5), 7 * 10**17);  // Threshold 0.7
        _registerConcept("creative", _generateRandomNeurons(5, 10), 7 * 10**17); // Threshold 0.7
        _registerConcept("analytical", _generateRandomNeurons(5, 10), 7 * 10**17); // Threshold 0.7
        _registerConcept("memory", _generateRandomNeurons(5, 15), 7 * 10**17);    // Threshold 0.7
        _registerConcept("emotional", _generateRandomNeurons(5, 20), 7 * 10**17);  // Threshold 0.7
        _registerConcept("social", _generateRandomNeurons(5, 25), 7 * 10**17);     // Threshold 0.7.              _registerConcept("philosophical", _generateRandomNeurons(5, 30), 7 * 10**17); // Threshold 0.7
        }


    /**
         * @dev Register a new concept for neural processing
         * @param name Concept name
         * @param associatedNeurons Neurons associated with this concept
         * @param activationThreshold Threshold for concept activation
         */
        function _registerConcept(
            string memory name,
            uint16[] memory associatedNeurons,
            uint256 activationThreshold
        ) internal {
            uint256 conceptId = conceptCount++;

            concepts[conceptId] = Concept({
                name: name,
                associatedNeurons: associatedNeurons,
                activationThreshold: activationThreshold,
                lastActivated: 0
            });
        }

        /**
         * @dev Generate random neuron IDs for concept association
         * @param count Number of neurons to generate
         * @param offset Starting offset for neuron IDs
         */
        function _generateRandomNeurons(uint16 count, uint16 offset) internal view returns (uint16[] memory) {
            uint16[] memory neuronIds = new uint16[](count);

            for (uint16 i = 0; i < count; i++) {
                // Generate semi-random neuron ID based on concept offset
                neuronIds[i] = uint16(offset + i + (uint256(keccak256(abi.encodePacked(block.timestamp, i, offset))) % 5));
            }

            return neuronIds;
        }

        /**
         * @dev Register a new concept manually (admin function)
         * @param name Concept name
         * @param neuronIds Neurons associated with this concept
         * @param activationThreshold Threshold for concept activation
         */
        function registerConcept(
            string memory name,
            uint16[] memory neuronIds,
            uint256 activationThreshold
        ) public onlyOwner returns (uint256) {
            uint256 conceptId = conceptCount++;

            concepts[conceptId] = Concept({
                name: name,
                associatedNeurons: neuronIds,
                activationThreshold: activationThreshold,
                lastActivated: 0
            });

            return conceptId;
        }

        /**
         * @dev Set the base URI for off-chain metadata
         * @param baseURI_ New base URI
         */
        function setBaseURI(string memory baseURI_) public onlyOwner {
            _baseURIExtended = baseURI_;
        }

        /**
         * @dev Override base URI for token URIs
         */
        function _baseURI() internal view virtual override returns (string memory) {
            return _baseURIExtended;
        }

        /**
         * @dev Toggle between on-chain and off-chain metadata
         * @param useOnChain True for on-chain metadata, false for off-chain
         */
        function setMetadataType(bool useOnChain) public onlyOwner {
            useOnChainMetadata = useOnChain;
        }

        /**
         * @dev Mints a new memory NFT with neural processing
         * @param to Recipient of the NFT
         * @param memoryText The text content of the memory
         * @param username Creator's username
         * @param qualityScore AI-evaluated quality score
         * @param rewardAmount Amount rewarded
         * @param tags Array of tags associated with the memory
         * @param xPostUrl URL to X (Twitter) post
         */
        function mintMemory(
            address to,
            string memory memoryText,
            string memory username,
            uint256 qualityScore,
            uint256 rewardAmount,
            string[] memory tags,
            string memory xPostUrl
        ) public onlyAuthorizedMinter returns (uint256) {
            require(bytes(memoryText).length > 0, "Memory text cannot be empty");
            require(bytes(username).length > 0, "Username cannot be empty");
            require(to != address(0), "Cannot mint to zero address");

            // Get current token ID and increment counter
            uint256 tokenId = _tokenIdCounter.current();
            _tokenIdCounter.increment();

            // Mint the NFT
            _safeMint(to, tokenId);

            // Process memory through neural network
            NeuralEncoding memory encoding = _processMemoryThroughSNN(memoryText, tokenId);

            // Store metadata
            memoryMetadata[tokenId] = MemoryMetadata({
                memoryText: memoryText,
                username: username,
                userWallet: to,
                qualityScore: qualityScore,
                rewardAmount: rewardAmount,
                timestamp: block.timestamp,
                tags: tags,
                xPostUrl: xPostUrl,
                neuralEncoding: encoding
            });

            // Store reference to memory's neural fingerprint
            neuralFingerprints[encoding.memoryFingerprint] = tokenId;

            emit MemoryMinted(tokenId, to, username, block.timestamp);
            return tokenId;
        }

        /**
         * @dev Process memory text through the Spiking Neural Network
         * @param memoryText The text content to process
         * @param tokenId The token ID for this memory
         * @return encoding Neural encoding of the memory
         */
        function _processMemoryThroughSNN(
            string memory memoryText,
            uint256 tokenId
        ) internal returns (NeuralEncoding memory) {
            // Convert text to neural inputs (simplified for demonstration)
            int256[] memory inputs = _textToNeuralInputs(memoryText);

            // Set inputs to the network
            snnContract.setNetworkInputs(globalSNNId, inputs);

            // Run multiple simulation steps
            for (uint i = 0; i < 10; i++) {
                snnContract.simulateTimeStep(globalSNNId);
            }

            // Capture activated neurons
            OnChainSNN.Spike[] memory spikes = snnContract.getRecentSpikes(globalSNNId, 50);

            // Convert spikes to activated neuron list
            uint16[] memory activatedNeurons = new uint16[](spikes.length);
            for (uint i = 0; i < spikes.length; i++) {
                activatedNeurons[i] = spikes[i].neuronId;
            }

            // Get output values for concept activation
            int256[] memory outputs = snnContract.getNetworkOutputs(globalSNNId);

            // Calculate concept activations
            int256[] memory conceptActivations = _calculateConceptActivations(activatedNeurons, outputs);

            // Generate memory fingerprint
            bytes32 memoryFingerprint = keccak256(abi.encodePacked(
                memoryText,
                activatedNeurons,
                conceptActivations,
                block.timestamp
            ));

            // Create and return neural encoding
            NeuralEncoding memory encoding = NeuralEncoding({
                networkId: globalSNNId,
                activatedNeurons: activatedNeurons,
                conceptActivations: conceptActivations,
                lastProcessed: block.timestamp,
                neuroplasticityScore: _calculateNeuroplasticityScore(activatedNeurons),
                memoryFingerprint: memoryFingerprint
            });

            emit MemoryProcessed(tokenId, globalSNNId, uint16(activatedNeurons.length));

            // Emit concept activation events
            for (uint i = 0; i < conceptActivations.length; i++) {
                    if (conceptActivations[i] >= int256(concepts[i].activationThreshold)) {
                    emit ConceptActivated(i, tokenId, uint256(conceptActivations[i]));
                    concepts[i].lastActivated = block.timestamp;
                }
            }

            return encoding;
        }

        /**
         * @dev Converts memory text to neural inputs
         * @param text The text to convert
         * @return inputs Array of neural input values
         */
        function _textToNeuralInputs(string memory text) internal pure returns (int256[] memory) {
            // For simplicity, we'll use the first 16 characters and convert them to inputs
            bytes memory textBytes = bytes(text);
            int256[] memory inputs = new int256[](16);

            for (uint i = 0; i < 16; i++) {
                if (i < textBytes.length) {
                    // Convert character to value between 0 and 1 (fixed point)
                    uint8 charVal = uint8(textBytes[i]);
                    // Scale to 0-1 range (assuming ASCII values 32-126)
                inputs[i] = int256(uint256(((charVal - 32) * 10**18) / 94));
                } else {
                    inputs[i] = 0; // Padding with 0 for shorter texts
                }
            }

            return inputs;
        }

        /**
         * @dev Calculate concept activations based on neural activity
         * @param activatedNeurons List of neurons that activated during processing
         * @param outputs Network output values
         * @return activations Array of concept activation levels
         */
        function _calculateConceptActivations(
            uint16[] memory activatedNeurons,
            int256[] memory outputs
        ) internal view returns (int256[] memory) {
            int256[] memory activations = new int256[](conceptCount);

            // Calculate activation for each concept
            for (uint i = 0; i < conceptCount; i++) {
                Concept storage concept = concepts[i];

                // Calculate overlap between activated neurons and concept neurons
                uint256 overlapCount = 0;
                for (uint j = 0; j < activatedNeurons.length; j++) {
                    for (uint k = 0; k < concept.associatedNeurons.length; k++) {
                        if (activatedNeurons[j] == concept.associatedNeurons[k]) {
                            overlapCount++;
                            break;
                        }
                    }
                }

                // Calculate activation based on overlap and output values
                int256 activation = 0;
                if (concept.associatedNeurons.length > 0) {
                    activation = int256((overlapCount * 10**18) / concept.associatedNeurons.length);

                    // Add component from output values (using first output for simplicity)
                    if (outputs.length > 0) {
                        activation = (activation * 8 + outputs[0] * 2) / 10; // 80% overlap, 20% output
                    }
                }

                activations[i] = activation;
            }

            return activations;
        }

        /**
         * @dev Calculate neuroplasticity score based on neural activity
         * @param activatedNeurons List of neurons that activated during processing
         * @return score Neuroplasticity score (0-100)
         */
        function _calculateNeuroplasticityScore(uint16[] memory activatedNeurons) internal pure returns (uint256) {
            // For simplicity, base score on number of activated neurons
            if (activatedNeurons.length == 0) return 0;

            // More activated neurons = higher plasticity (up to a point)
            uint256 score = activatedNeurons.length * 10;
            if (score > 100) score = 100;

            return score;
        }

        /**
         * @dev Reprocess a memory through the neural network
         * @param tokenId The token ID to reprocess
         */
        function reprocessMemory(uint256 tokenId) public {
            require(_exists(tokenId), "Token does not exist");

            MemoryMetadata storage metadata = memoryMetadata[tokenId];

            // Process memory through neural network again
            NeuralEncoding memory newEncoding = _processMemoryThroughSNN(
                metadata.memoryText,
                tokenId
            );

            // Update neural encoding
            metadata.neuralEncoding = newEncoding;

            // Update neural fingerprint reference
            neuralFingerprints[newEncoding.memoryFingerprint] = tokenId;
        }

        /**
         * @dev Find semantically similar memories based on neural patterns
         * @param tokenId The reference token ID
         * @param maxResults Maximum number of results to return
         * @return similarTokenIds Array of similar token IDs
         * @return similarityScores Array of similarity scores
         */
        function findSimilarMemories(
            uint256 tokenId,
            uint8 maxResults
        ) public view returns (uint256[] memory, uint256[] memory) {
            require(_exists(tokenId), "Token does not exist");
            require(maxResults > 0, "Max results must be greater than 0");

            // Get reference encoding
            NeuralEncoding storage refEncoding = memoryMetadata[tokenId].neuralEncoding;

            // Array to store similarity scores
            uint256[] memory scores = new uint256[](_tokenIdCounter.current());
            uint256[] memory tokenIds = new uint256[](_tokenIdCounter.current());
            uint256 count = 0;

            // Calculate similarity scores for all tokens
            for (uint256 i = 0; i < _tokenIdCounter.current(); i++) {
                if (i != tokenId && _exists(i)) {
                    NeuralEncoding storage encoding = memoryMetadata[i].neuralEncoding;

                    // Calculate similarity score
                    uint256 score = _calculateNeuralSimilarity(refEncoding, encoding);

                    // Store score and token ID
                    scores[count] = score;
                    tokenIds[count] = i;
                    count++;
                }
            }

            // Sort results by similarity score (descending)
            for (uint256 i = 0; i < count; i++) {
                for (uint256 j = i + 1; j < count; j++) {
                    if (scores[j] > scores[i]) {
                        // Swap scores
                        uint256 tempScore = scores[i];
                        scores[i] = scores[j];
                        scores[j] = tempScore;

                        // Swap token IDs
                        uint256 tempId = tokenIds[i];
                        tokenIds[i] = tokenIds[j];
                        tokenIds[j] = tempId;
                    }
                }
            }

            // Limit results
            uint256 resultCount = count < maxResults ? count : maxResults;

            // Create result arrays
            uint256[] memory resultIds = new uint256[](resultCount);
            uint256[] memory resultScores = new uint256[](resultCount);

            // Copy top results
            for (uint256 i = 0; i < resultCount; i++) {
                resultIds[i] = tokenIds[i];
                resultScores[i] = scores[i];
            }

            return (resultIds, resultScores);
        }

        /**
         * @dev Calculate neural similarity between two memory encodings
         * @param encoding1 First encoding
         * @param encoding2 Second encoding
         * @return score Similarity score (0-100)
         */
        function _calculateNeuralSimilarity(
            NeuralEncoding storage encoding1,
            NeuralEncoding storage encoding2
        ) internal view returns (uint256) {
            // Calculate overlap between activated neurons
            uint256 neuronOverlap = 0;
            for (uint i = 0; i < encoding1.activatedNeurons.length; i++) {
                for (uint j = 0; j < encoding2.activatedNeurons.length; j++) {
                    if (encoding1.activatedNeurons[i] == encoding2.activatedNeurons[j]) {
                        neuronOverlap++;
                        break;
                    }
                }
            }

            // Calculate concept activation similarity
            uint256 conceptSimilarity = 0;
            if (encoding1.conceptActivations.length == encoding2.conceptActivations.length) {
                uint256 totalDifference = 0;
                for (uint i = 0; i < encoding1.conceptActivations.length; i++) {
                    // Calculate absolute difference between concept activations
                    int256 diff = encoding1.conceptActivations[i] - encoding2.conceptActivations[i];
                    if (diff < 0) diff = -diff;

                    // Sum differences
                    totalDifference += uint256(diff);
                }

                // Convert to similarity score (0-100)
                if (encoding1.conceptActivations.length > 0) {
                    uint256 avgDiff = totalDifference / encoding1.conceptActivations.length;
                    // Lower difference = higher similarity
                    conceptSimilarity = avgDiff < 10**18 ? 100 - (avgDiff * 100 / 10**18) : 0;
                }
            }

            // Combine neuron overlap and concept similarity
            uint256 neuronSimilarity = 0;
            if (encoding1.activatedNeurons.length > 0 && encoding2.activatedNeurons.length > 0) {
                neuronSimilarity = (neuronOverlap * 100) / 
                    ((encoding1.activatedNeurons.length + encoding2.activatedNeurons.length) / 2);
            }

            // Final similarity score (70% concept, 30% neuron)
            return (conceptSimilarity * 70 + neuronSimilarity * 30) / 100;
        }

        /**
         * @dev Get neural encoding for a memory
         * @param tokenId The token ID
         * @return networkId SNN network ID
         * @return activatedNeurons Neurons activated by this memory
         * @return neuroplasticityScore Memory's neuroplasticity score
         */
        function getNeuralEncoding(uint256 tokenId) public view returns (
            uint256 networkId,
            uint16[] memory activatedNeurons,
            uint256 neuroplasticityScore
        ) {
            require(_exists(tokenId), "Token does not exist");
            NeuralEncoding storage encoding = memoryMetadata[tokenId].neuralEncoding;

            return (
                encoding.networkId,
                encoding.activatedNeurons,
                encoding.neuroplasticityScore
            );
        }

        /**
         * @dev Get concept activations for a memory
         * @param tokenId The token ID
         * @return conceptNames Names of all concepts
         * @return activationLevels Activation levels for each concept
         */
        function getConceptActivations(uint256 tokenId) public view returns (
            string[] memory conceptNames,
            int256[] memory activationLevels
        ) {
            require(_exists(tokenId), "Token does not exist");
            NeuralEncoding storage encoding = memoryMetadata[tokenId].neuralEncoding;

            // Create arrays for concept names and activation levels
            string[] memory names = new string[](conceptCount);
            int256[] memory activations = new int256[](conceptCount);

            // Populate arrays
            for (uint i = 0; i < conceptCount; i++) {
                names[i] = concepts[i].name;
                activations[i] = i < encoding.conceptActivations.length ? 
                 encoding.conceptActivations[i] : int256(0);
            }

            return (names, activations);
        }

        /**
         * @dev Generates JSON metadata for a token
         * @param tokenId The ID of the token
         */
        function _generateMetadataJSON(uint256 tokenId) internal view returns (string memory) {
            MemoryMetadata storage metadata = memoryMetadata[tokenId];

            // Generate neural attributes
            string memory neuralAttributes = string(abi.encodePacked(
                '{"trait_type":"Neuroplasticity Score","value":', 
                uint256(metadata.neuralEncoding.neuroplasticityScore).toString(), '},'
            ));

            // Add concept activations
            (string[] memory conceptNames, int256[] memory activationLevels) = getConceptActivations(tokenId);
            for (uint i = 0; i < conceptNames.length; i++) {
                if (activationLevels[i] > 0) {
                    neuralAttributes = string(abi.encodePacked(
                        neuralAttributes,
                        '{"trait_type":"Concept:', conceptNames[i], '","value":', 
                        uint256(uint256(activationLevels[i]) / 10**16).toString(), '},'
                    ));
                }
            }

            // Generate the JSON attributes array
            string memory attributes = string(abi.encodePacked(
                '[{"trait_type":"Username","value":"', metadata.username, '"},',
                '{"trait_type":"Quality Score","value":', metadata.qualityScore.toString(), '},',
                '{"trait_type":"Reward Amount","value":', metadata.rewardAmount.toString(), '},',
                '{"trait_type":"Timestamp","value":', metadata.timestamp.toString(), '},',
                neuralAttributes
            ));

            // Add X post URL if available
            if (bytes(metadata.xPostUrl).length > 0) {
                attributes = string(abi.encodePacked(
                    attributes, '{"trait_type":"X Post URL","value":"', metadata.xPostUrl, '"},'
                ));
            }

            // Add tags
            string memory tagsValue = "";
            if (metadata.tags.length > 0) {
                tagsValue = metadata.tags[0];
                for (uint i = 1; i < metadata.tags.length; i++) {
                    tagsValue = string(abi.encodePacked(tagsValue, ", ", metadata.tags[i]));
                }
            }
            attributes = string(abi.encodePacked(attributes, '{"trait_type":"Tags","value":"', tagsValue, '"}]'));

            // Create the final JSON
            string memory json = Base64.encode(bytes(string(abi.encodePacked(
                '{"name":"NeuralSOCIAL Memory #', tokenId.toString(), '",',
                '"description":"', metadata.memoryText, '",',
                '"attributes":', attributes,
                (bytes(metadata.xPostUrl).length > 0 ? string(abi.encodePacked(',"external_url":"', metadata.xPostUrl, '"')) : ""),
                '}'
            ))));

            return string(abi.encodePacked('data:application/json;base64,', json));
        }

        /**
         * @dev Override for tokenURI to provide proper metadata (on-chain or off-chain)
         * @param tokenId The ID of the token
         */
        function tokenURI(uint256 tokenId) public view override returns (string memory) {
            require(_exists(tokenId), "Token does not exist");

            if (useOnChainMetadata) {
                // Return on-chain base64 encoded JSON
                return _generateMetadataJSON(tokenId);
            } else {
                // Return off-chain URI (baseURI + tokenId)
                return string(abi.encodePacked(_baseURI(), tokenId.toString()));
            }
        }

        /**
         * @dev Get total supply of memories minted
         */
        function totalSupply() public view returns (uint256) {
            return _tokenIdCounter.current();
        }
    }
