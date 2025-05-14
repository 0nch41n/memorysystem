# NeuraMint: Onchain Neural Network NFT System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

NeuraMint is an onchain spiking neural network (SNN) system integrated with NFT technology. It creates neural representations of content, enabling semantic similarity search, concept detection, and adaptive learning through neuroplasticity.

## 🧠 Overview

NeuraMint consists of two main contracts:

1. **OnChainSNN** - A fully on-chain implementation of a spiking neural network with learning capabilities
2. **NeuralSOCIAL** - An ERC721 NFT contract that integrates with the neural network to create memory tokens

The system processes text-based memories through the neural network, creating unique neural fingerprints that capture semantic meaning. These neural representations enable concept detection and similarity matching between memories.

## ✨ Key Features

- **On-Chain Neural Network**: Fully functional spiking neural network implementation in Solidity
- **Neuroplasticity**: Learning through Spike-Timing-Dependent Plasticity (STDP)
- **Neural Fingerprinting**: Create unique neural signatures for content
- **Concept Detection**: Identify abstract concepts in content through neural activation
- **Semantic Search**: Find similar memories based on neural patterns
- **On-Chain Metadata**: Rich NFT metadata including neural attributes

## 🔧 Technical Components

### OnChainSNN Contract

```solidity
contract OnChainSNN is Ownable {
    // Fixed-point arithmetic precision (18 decimals)
    int256 constant FP_SCALING = 10**18;
    // ...
}
```

- **Neuron Types**: Input, Excitatory, and Inhibitory neurons
- **Network Architecture**: Supports input, hidden, and output layers
- **Synaptic Connections**: Weighted connections between neurons
- **Membrane Potential**: Simulates neuron activation dynamics
- **Learning**: Implements STDP for weight adjustments

### NeuralSOCIAL Contract

```solidity
contract NeuralSOCIAL is ERC721, Ownable {
    // Neural encoding for memories
    struct NeuralEncoding {
        uint256 networkId;
        uint16[] activatedNeurons;
        int256[] conceptActivations;
        // ...
    }
    // ...
}
```

- **Memory Encoding**: Process text through the neural network
- **Concept Registry**: Map neural patterns to abstract concepts
- **Memory Similarity**: Calculate neural similarity between memories
- **NFT Integration**: Represent memories as NFTs with neural attributes

## 📊 Use Cases

- **Semantic Memory Storage**: Store memories and find connections between them
- **Content Recommendation**: Discover related content based on neural similarity
- **Concept Detection**: Identify abstract themes in content
- **Evolving Collections**: NFT collections that learn and adapt over time
- **On-Chain Intelligence**: Foundation for more complex AI systems on blockchain

## 🔍 Technical Deep Dive

### Fixed-Point Arithmetic

The system uses 18-decimal place fixed-point arithmetic to simulate floating-point operations in Solidity:

```solidity
// Example of exponential decay function
function _exponentialDecay(int256 x, int256 tau) internal pure returns (int256) {
    // Taylor series approximation for e^(-x/tau)
    // ...
}
```

### Neural Learning

The STDP learning rule adjusts synaptic weights based on the timing of neuron activations:

```solidity
// Calculate weight delta for STDP
function _calculateWeightDelta(int256 timeDiff, bool isLTP) internal pure returns (int256) {
    // Simple STDP rule - weight change decreases exponentially with time difference
    int256 maxChange = isLTP ? FP_SCALING / 10 : -(FP_SCALING / 20); // LTP stronger than LTD
    // ...
}
```

### Neural Fingerprinting

Each memory is processed through the neural network to create a unique fingerprint:

```solidity
// Generate memory fingerprint
bytes32 memoryFingerprint = keccak256(abi.encodePacked(
    memoryText,
    activatedNeurons,
    conceptActivations,
    block.timestamp
));
```

### Deployment

1. Deploy the OnChainSNN contract first
   ```bash
   npx hardhat run scripts/deploy-snn.js --network <your-network>
   ```

2. Deploy the NeuralSOCIAL contract with the SNN contract address
   ```bash
   npx hardhat run scripts/deploy-neural-social.js --network <your-network>
   ```

### Example Usage

```javascript
// Initialize the neural network
await neuralSocial.mintMemory(
    userAddress,
    "This is a memory about artificial intelligence and blockchain technology.",
    "username",
    85, // quality score
    100, // reward amount
    ["AI", "blockchain"], // tags
    "https://x.com/post/123" // X post URL
);

// Find similar memories
const [similarIds, scores] = await neuralSocial.findSimilarMemories(tokenId, 5);
```

## 📈 Gas Optimization

The system implements several gas optimization strategies:
- Optimized data types (uint8, uint16, etc.)
- Efficient storage of neural activations
- Limited spike history storage
- Careful memory management

## 🔒 Security Considerations

- Neural networks can be manipulated through careful input crafting
- Fixed-point arithmetic requires careful bounds checking
- Learning rules may lead to unexpected behavior over time

## 🧪 Advanced Experimentation

### Custom Concepts

Register your own concepts to detect in memories:

```solidity
uint16[] memory associatedNeurons = new uint16[](5);
// Assign neurons associated with the concept
associatedNeurons[0] = 18;
associatedNeurons[1] = 24;
// ...

await neuralSocial.registerConcept(
    "technology",
    associatedNeurons,
    7 * 10**17 // 0.7 activation threshold
);
```

### Network Visualization

Extract network structure and activations for visualization:

```javascript
const [networkId, activatedNeurons, neuroplasticityScore] = 
    await neuralSocial.getNeuralEncoding(tokenId);

const [conceptNames, activationLevels] = 
    await neuralSocial.getConceptActivations(tokenId);
```

## 📚 Further Reading

- [Spiking Neural Networks: A Primer](https://en.wikipedia.org/wiki/Spiking_neural_network)
- [Spike-Timing-Dependent Plasticity](https://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity)
- [Fixed-Point Arithmetic in Solidity](https://ethereum.org/en/developers/docs/smart-contracts/developer-considerations/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- OpenZeppelin for secure contract implementations
- The Ethereum community for ongoing innovation in smart contracts
- Neuroscience research that inspired the neural network models
