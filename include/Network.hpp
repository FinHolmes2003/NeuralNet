#pragma once

#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <sstream>

namespace Network
{
    using json = nlohmann::json;
    using std::vector;
    using std::pair;
    using std::string;
    using std::function;
    using std::invalid_argument;
    using std::tuple;

    /// @brief Fast Sigmoid Function
    /// @param inputValue Raw input value
    /// @return Activated value bounded between -1 and 1
    inline float Softsign(const float inputValue)
    {
        return inputValue / (1.0f + std::abs(inputValue));
    }

    /// @brief Fast reverse Sigmoid derivative for backpropagation
    /// @param inputValue Input value processed by activation
    /// @return Calculated derivative value
    inline float InverseSoftsign(const float inputValue)
    {
        float denom = 1.0f + std::abs(inputValue);
        return 1.0f / (denom * denom);
    }

    /// @brief Rectified Linear Unit function
    /// @param inputValue Input tensor value
    /// @return Bounded float output (max between 0 and value)
    inline float ReLU(const float inputValue)
    {
        return inputValue > 0.0f ? inputValue : 0.0f;
    }

    /// @brief Derivative of ReLU for backpropagation
    /// @param inputValue Input tensor value
    /// @return 1.0f if positive, 0.0f otherwise
    inline float InverseReLU(const float inputValue)
    {
        return inputValue > 0.0f ? 1.0f : 0.0f;
    }

    /// @enum OptimizerType
    /// @brief Supported optimization strategies for weight updates
    enum class OptimizerType
    {
        SGD,
        AdamW 
    };

    /// @enum ActivationType
    /// @brief Supported built-in and custom activation function identifiers
    enum class ActivationType
    {
        Softsign,
        ReLU,
        Custom
    };

    /// @brief Changes the Activation function type to a string for JSON serialization
    /// @param activation Enum identifying the activation function type
    /// @return String representation of the activation name
    inline string ActivationTypeToName(const ActivationType& activation)
    {
        switch (activation)
        {
            case ActivationType::Softsign: return "Softsign";
            case ActivationType::ReLU:     return "ReLU";
            default:                       return "Custom";
        }
    }

    /// @brief Parses an activation function string name back into its corresponding Enum
    /// @param name String identifier from JSON
    /// @return Enum representation of activation function type
    inline ActivationType ActivationNameToType(const string& name)
    {
        if (name == "Softsign") return ActivationType::Softsign;
        if (name == "ReLU")     return ActivationType::ReLU;
        return ActivationType::Custom;
    }

    /// @brief Retrieves the activation and derivative functions for the given Enum
    /// @param type Activation function type enum
    /// @return Pair containing {ActivationFunction, DerivativeFunction}
    inline pair<function<float(float)>, function<float(float)>> GetActivationFunctions(const ActivationType& type)
    {
        switch (type)
        {
            case ActivationType::ReLU: return { ReLU, InverseReLU };
            case ActivationType::Softsign:
            default: return { Softsign, InverseSoftsign };
        }
    }

    /// @brief Deserializes network layers from JSON into raw arrays
    /// @param weightsFile Parsed JSON containing network parameters
    /// @return Tuple containing {Layer Biases, Layer Weights, Layer Activation Enums}
    inline tuple<vector<vector<float>>, vector<vector<float>>, vector<ActivationType>> JsonToVectors(const json& weightsFile)
    {
        size_t numLayers = weightsFile["Layers"].size();

        vector<vector<float>> biases(numLayers);
        vector<vector<float>> weights(numLayers);
        vector<ActivationType> activations(numLayers);

        for (size_t i = 0; i < numLayers; i++)
        {
            biases[i] = weightsFile["Layers"][i]["Biases"].get<vector<float>>();

            activations[i] = weightsFile["Layers"][i].contains("ActivationFunction")
                ? ActivationNameToType(weightsFile["Layers"][i]["ActivationFunction"].get<string>())
                : ActivationType::Softsign;
            
            auto weights2D = weightsFile["Layers"][i]["Weights"].get<vector<vector<float>>>();
            size_t numInputs = weights2D.size();
            size_t numOutputs = numInputs == 0 ? 0 : weights2D[0].size();

            weights[i].resize(numInputs * numOutputs);
            for (size_t row = 0; row < numInputs; ++row)
            {
                for (size_t col = 0; col < numOutputs; ++col)
                {
                    weights[i][row * numOutputs + col] = weights2D[row][col];
                }
            }
        }

        return {biases, weights, activations};
    }

    class Network
    {
        public:
            /// @brief Callback payload for training progress updates
            /// @param currentEpoch Current training epoch step
            /// @param totalEpochs Total target epochs
            /// @param currentBatch Index of the executing batch
            /// @param totalBatch Total batches per epoch
            /// @param trainingSample Number of samples evaluated
            /// @param trainingSampleNum Total training dataset size
            /// @param currentLoss Calculated epoch loss value
            using ProgressCallback = std::function<void(int currentEpoch, int totalEpochs, int currentBatch, int totalBatch, int trainingSample, int trainingSampleNum, float currentLoss)>;

            /// @struct Layer
            /// @brief Represents a single fully connected network layer with cache and momentum state
            struct Layer
            {
                /// @brief Number of incoming input connections
                size_t numInputs = 0;

                /// @brief Number of outgoing output nodes
                size_t numOutputs = 0;

                /// @brief Current bias parameters for output nodes
                vector<float> biases;

                /// @brief Accumulated bias gradients across mini-batches
                vector<float> biasGradients;

                /// @brief AdamW first moment vectors for biases
                vector<float> mBiases;

                /// @brief AdamW second moment vectors for biases
                vector<float> vBiases;

                /// @brief Current flattened weight parameters
                vector<float> weights;

                /// @brief Accumulated weight gradients across mini-batches
                vector<float> weightGradients;

                /// @brief AdamW first moment vectors for weights
                vector<float> mWeights;

                /// @brief AdamW second moment vectors for weights
                vector<float> vWeights;

                /// @brief Bound forward activation callback function
                function<float(float)> activationFunction;

                /// @brief Bound derivative activation callback function
                function<float(float)> invActivationFunction;

                /// @brief Enum flag representing active activation strategy
                ActivationType activationType = ActivationType::Softsign;

                /// @brief Cached input vector from previous forward pass
                vector<float> lastInputs;

                /// @brief Cached pre-activation sums from previous forward pass
                vector<float> lastSums;

                /// @brief Cached activated outputs from previous forward pass
                vector<float> lastOutputs;

                /// @brief Step counter variable for AdamW bias corrections
                size_t adamStep = 0;

                /// @brief Constructor initializing layer dimensions and functions
                /// @param numInputs Incoming node count
                /// @param numOutputs Outgoing node count
                /// @param actType Enum type for activation
                /// @param actvFunc Custom forward activation implementation (optional)
                /// @param invActvFunc Custom derivative activation implementation (optional)
                Layer(const size_t numInputs = 0, const size_t numOutputs = 0, const ActivationType actType = ActivationType::Softsign,
                      const function<float(float)> actvFunc = nullptr, const function<float(float)> invActvFunc = nullptr)
                        : numInputs(numInputs), numOutputs(numOutputs), activationType(actType)
                {
                    biases.resize(numOutputs, 0.0f);
                    biasGradients.resize(numOutputs, 0.0f);
                    mBiases.resize(numOutputs, 0.0f);
                    vBiases.resize(numOutputs, 0.0f);

                    size_t totalWeights = numInputs * numOutputs;
                    weights.resize(totalWeights);
                    weightGradients.resize(totalWeights, 0.0f);
                    mWeights.resize(totalWeights, 0.0f);
                    vWeights.resize(totalWeights, 0.0f);

                    auto funcs = GetActivationFunctions(actType);
                    activationFunction = funcs.first;
                    invActivationFunction = funcs.second;

                    float limit = (numInputs > 0) ? std::sqrt(2.0f / static_cast<float>(numInputs)) : 1.0f;

                    for (size_t k = 0; k < totalWeights; k++)
                    {
                        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                        weights[k] = r * (2.0f * limit) - limit;
                    }

                    if (actType == ActivationType::Custom)
                    {
                        if (actvFunc == nullptr || invActvFunc == nullptr)
                        {
                            throw std::invalid_argument("If using a custom activation function, please input both functions");
                        }
                        activationFunction = actvFunc;
                        invActivationFunction = invActvFunc;
                    }
                }

                /// @brief Constructor initializing layer with explicit weights and biases
                /// @param biasValues Initial bias values array
                /// @param weightValues Initial flattened weight array
                /// @param actType Enum identifier for activation
                /// @param actvFunc Custom activation function target (optional)
                /// @param invActFunc Custom derivative activation target (optional)
                Layer(const vector<float>& biasValues, const vector<float>& weightValues, ActivationType actType = ActivationType::Softsign,
                      const function<float(float)> actvFunc = nullptr, const function<float(float)> invActFunc = nullptr)
                        : biases(biasValues), weights(weightValues), activationType(actType)
                {
                    numOutputs = biases.size();
                    numInputs = numOutputs == 0 ? 0 : weights.size() / numOutputs;

                    biasGradients.resize(numOutputs, 0.0f);
                    mBiases.resize(numOutputs, 0.0f);
                    vBiases.resize(numOutputs, 0.0f);

                    weightGradients.resize(weights.size(), 0.0f);
                    mWeights.resize(weights.size(), 0.0f);
                    vWeights.resize(weights.size(), 0.0f);

                    auto funcs = GetActivationFunctions(actType);
                    activationFunction = funcs.first;
                    invActivationFunction = funcs.second;

                    if (actType == ActivationType::Custom)
                    {
                        if (actvFunc == nullptr || invActFunc == nullptr)
                        {
                            throw std::invalid_argument("If using a custom activation function, please input both functions");
                        }
                        activationFunction = actvFunc;
                        invActivationFunction = invActFunc;
                    }
                }

                /// @brief Serializes current layer weights and state into JSON format
                /// @return nlohmann::json structured output representation
                json CreateJsonLayer() const
                {
                    vector<vector<float>> weights2D(numInputs, vector<float>(numOutputs));
                    for (size_t i = 0; i < numInputs; ++i)
                    {
                        for (size_t j = 0; j < numOutputs; ++j)
                        {
                            weights2D[i][j] = weights[i * numOutputs + j];
                        }
                    }

                    string activationFuncName = ActivationTypeToName(activationType);
                    return json{
                        {"Biases", biases},
                        {"Weights", weights2D},
                        {"ActivationFunction", activationFuncName}
                    };
                }

                /// @brief Updates current layer biases and resets cached gradients
                /// @param newBiases New bias vector array
                void UpdateBiases(const vector<float>& newBiases)
                {
                    biases = newBiases;
                    numOutputs = biases.size();
                    biasGradients.assign(numOutputs, 0.0f);
                }

                /// @brief Updates current layer weights and recalculates dimensions
                /// @param newWeights Flattened replacement weights array
                void UpdateWeights(const vector<float>& newWeights)
                {
                    weights = newWeights;
                    numInputs = numOutputs == 0 ? 0 : weights.size() / numOutputs;
                    weightGradients.assign(weights.size(), 0.0f);
                }

                /// @brief Evaluates node activations across input features
                /// @param inputs Input feature values for this layer
                /// @return Reference to calculated output activation vector
                const vector<float>& ForwardPass(const vector<float>& inputs)
                {
                    if (numOutputs == 0) return inputs;

                    lastInputs = inputs; 
                    lastSums.assign(numOutputs, 0.0f);
                    lastOutputs.resize(numOutputs);

                    for (size_t j = 0; j < numOutputs; ++j)
                    {
                        lastSums[j] = biases[j];
                    }

                    for (size_t i = 0; i < numInputs; ++i)
                    {
                        float inVal = inputs[i];
                        size_t rowOffset = i * numOutputs;

                        for (size_t j = 0; j < numOutputs; ++j)
                        {
                            lastSums[j] += inVal * weights[rowOffset + j];
                        }
                    }

                    for (size_t j = 0; j < numOutputs; ++j)
                    {
                        lastOutputs[j] = activationFunction(lastSums[j]);
                    }

                    return lastOutputs;
                }

                /// @brief Accumulates parameter gradients and calculates error propagation deltas
                /// @param deltas Gradient error components from downstream layer
                /// @return Accumulated error deltas mapped for upstream layer
                vector<float> AccumulateGradients(const vector<float>& deltas)
                {
                    vector<float> prevDeltas(numInputs, 0.0f);

                    for (size_t i = 0; i < numInputs; ++i)
                    {
                        float inputI = lastInputs[i];
                        size_t rowOffset = i * numOutputs;

                        for (size_t j = 0; j < numOutputs; ++j)
                        {
                            float deltaJ = deltas[j];

                            prevDeltas[i] += weights[rowOffset + j] * deltaJ;
                            weightGradients[rowOffset + j] += inputI * deltaJ;
                        }
                    }

                    for (size_t j = 0; j < numOutputs; ++j)
                    {
                        biasGradients[j] += deltas[j];
                    }

                    return prevDeltas;
                }

                /// @brief Applies parameter updates via Stochastic Gradient Descent (SGD)
                /// @param learningRate Current learning step size multiplier
                /// @param currentBatchSize Number of samples in accumulated mini-batch
                void ApplyGradients(const float& learningRate, const size_t& currentBatchSize)
                {
                    float invBatch = 1.0f / static_cast<float>(currentBatchSize);
                    size_t totalWeights = weights.size();

                    for (size_t k = 0; k < totalWeights; ++k)
                    {
                        float avgGrad = weightGradients[k] * invBatch;
                        weights[k] -= learningRate * avgGrad;
                        weightGradients[k] = 0.0f;
                    }

                    for (size_t j = 0; j < numOutputs; ++j)
                    {
                        float avgBiasGrad = biasGradients[j] * invBatch;
                        biases[j] -= learningRate * avgBiasGrad;
                        biasGradients[j] = 0.0f;
                    }
                }

                /// @brief Applies parameter updates via AdamW optimizer with decoupled decay
                /// @param learningRate Current step learning rate
                /// @param currentBatchSize Total mini-batch sample count
                /// @param beta1 Exponential decay multiplier for first moment estimation
                /// @param beta2 Exponential decay multiplier for second moment estimation
                /// @param eps Epsilon value for division stabilization
                /// @param weightDecay Decoupled weight decay multiplier
                void ApplyGradientsAdamW(const float& learningRate, const size_t& currentBatchSize, const float& beta1 = 0.9f,
                    const float& beta2 = 0.999f, const float& eps = 1e-8f, const float& weightDecay = 0.01f)
                {
                    if (currentBatchSize == 0) return;

                    adamStep++;
                    float invBatch = 1.0f / static_cast<float>(currentBatchSize);

                    float biasCorrection1 = 1.0f / (1.0f - std::pow(beta1, static_cast<float>(adamStep)));
                    float biasCorrection2 = 1.0f / (1.0f - std::pow(beta2, static_cast<float>(adamStep)));

                    size_t totalWeights = weights.size();
                    for (size_t k = 0; k < totalWeights; k++)
                    {
                        float g = weightGradients[k] * invBatch;

                        weights[k] -= learningRate * weightDecay * weights[k];

                        mWeights[k] = beta1 * mWeights[k] + (1.0f - beta1) * g;
                        vWeights[k] = beta2 * vWeights[k] + (1.0f - beta2) * (g * g);

                        float mHat = mWeights[k] * biasCorrection1;
                        float vHat = vWeights[k] * biasCorrection2;

                        weights[k] -= learningRate * mHat / (std::sqrt(vHat) + eps);
                        weightGradients[k] = 0.0f;
                    }

                    for (size_t j = 0; j < numOutputs; j++)
                    {
                        float g = biasGradients[j] * invBatch;

                        mBiases[j] = beta1 * mBiases[j] + (1.0f - beta1) * g;
                        vBiases[j] = beta2 * vBiases[j] + (1.0f - beta2) * (g * g);

                        float mHat = mBiases[j] * biasCorrection1;
                        float vHat = vBiases[j] * biasCorrection2;

                        biases[j] -= learningRate * mHat / (std::sqrt(vHat) + eps);
                        biasGradients[j] = 0.0f;
                    }
                }

                /// @brief Replaces active layer activation routines and updates internal type
                /// @param actvFuncType New activation type Enum
                /// @param actvFunc Custom activation callback function (optional)
                /// @param invActvFunc Custom derivative activation callback function (optional)
                void ChangeActivationFunction(const ActivationType& actvFuncType,
                    const function<float(float)>& actvFunc = nullptr,
                    const function<float(float)>& invActvFunc = nullptr)
                {
                    activationType = actvFuncType;
                    if (actvFuncType == ActivationType::Custom)
                    {
                        if (!actvFunc || !invActvFunc)
                            throw std::invalid_argument("If using a custom activation function, please input both functions");
                        activationFunction = actvFunc;
                        invActivationFunction = invActvFunc;
                    }
                    else
                    {
                        auto funcs = GetActivationFunctions(actvFuncType);
                        activationFunction = funcs.first;
                        invActivationFunction = funcs.second;
                    }
                }
            };

            /// @brief Constructor constructing model architecture based on layer node dimensions
            /// @param layerSizes Array containing neuron counts per layer {input, hidden..., output}
            /// @param actvFuncs Vector specifying ActivationType per layer step
            /// @param actvFunc Global fallback implementation for custom activation
            /// @param invActvFunc Global fallback implementation for custom derivative
            Network(const vector<int>& layerSizes,
                    vector<ActivationType> actvFuncs = {},
                    const function<float(float)>& actvFunc = nullptr,
                    const function<float(float)>& invActvFunc = nullptr)
            {
                layers.resize(layerSizes.size() - 1);

                if (actvFuncs.empty())
                {
                    actvFuncs.resize(layerSizes.size() - 1, ActivationType::Softsign);
                }

                for (size_t i = 0; i < layers.size(); i++)
                {
                    layers[i] = Layer(layerSizes[i], layerSizes[i + 1], actvFuncs[i], actvFunc, invActvFunc);
                }
            }

            /// @brief Constructor initializing model from explicit parameters
            /// @param biases Array of layer bias values
            /// @param weights Array of layer weight values
            /// @param activFuncs Vector specifying activation strategy per layer
            /// @param actvFunc Global fallback custom activation function
            /// @param invActvFunc Global fallback custom derivative activation function
            Network(const vector<vector<float>>& biases, const vector<vector<float>>& weights,
                    vector<ActivationType> activFuncs = {},
                    const function<float(float)>& actvFunc = nullptr,
                    const function<float(float)>& invActvFunc = nullptr)
            {
                if (biases.size() != weights.size())
                {
                    throw invalid_argument("Mismatch between number of bias layers and weight layers.");
                }

                if (activFuncs.empty())
                {
                    activFuncs.resize(biases.size(), ActivationType::Softsign);
                }

                layers.reserve(biases.size());
                for (size_t i = 0; i < biases.size(); i++)
                {
                    layers.emplace_back(biases[i], weights[i], activFuncs[i], actvFunc, invActvFunc);
                }
            }

            /// @brief Default empty Network constructor
            Network(){}

            /// @brief Accessor to read structural layer vector parameters
            /// @return Read-only constant reference to model layers
            const vector<Layer>& GetLayers() const { return layers; }

            /// @brief Formats network structural configuration for debugging output
            /// @return String representation displaying biases and layer weights
            string VisualNetwork() const
            {
                std::ostringstream output;

                for (size_t i = 0; i < layers.size(); ++i)
                {
                    const auto& layer = layers[i];
                    
                    for (size_t inIdx = 0; inIdx < layer.numInputs; ++inIdx)
                    {
                        size_t rowOffset = inIdx * layer.numOutputs;
                        
                        for (size_t outIdx = 0; outIdx < layer.numOutputs; ++outIdx)
                        {
                            output << layer.weights[rowOffset + outIdx] << " ";
                        }

                        if (inIdx != layer.numInputs - 1)
                        {
                            output << "| ";
                        }
                    }
                    output << "\n";

                    for (size_t outIdx = 0; outIdx < layer.numOutputs; ++outIdx)
                    {
                        output << layer.biases[outIdx] << " ";
                    }
                    output << "\n";
                }

                return output.str();
            }

            /// @brief Executes network inference from input vector to final output layer
            /// @param inputs Array of values matching input layer dimension
            /// @return Output vector containing forward prediction results
            const vector<float>& ForwardPass(const vector<float>& inputs)
            {
                if (layers.empty() || inputs.size() != layers[0].numInputs)
                {
                    throw invalid_argument("Input size mismatch with network input layer.");
                }

                const vector<float>* currentInputs = &inputs;
                for (size_t i = 0; i < layers.size(); ++i)
                {
                    currentInputs = &layers[i].ForwardPass(*currentInputs);
                }
                return *currentInputs;
            }

            /// @brief Trains network parameters using backpropagation and specified optimization routine
            /// @param trainingData Vector of sample pairs {Input features, Target labels}
            /// @param epochs Total number of dataset passes
            /// @param batchSize Mini-batch update sample threshold
            /// @param learningRate Base step learning rate factor
            /// @param onProgress Step callback logging system metrics (optional)
            /// @param optimizer Selected strategy choice (SGD or AdamW)
            /// @param beta1 First moment estimation decay (AdamW)
            /// @param beta2 Second moment estimation decay (AdamW)
            /// @param eps Variance numerical tolerance threshold (AdamW)
            /// @param weightDecay L2 decoupled weight regularization factor (AdamW)
            void Train(const vector<pair<vector<float>, vector<float>>>& trainingData, const int& epochs, const size_t& batchSize, const float& learningRate, 
                    const ProgressCallback& onProgress = nullptr, const OptimizerType& optimizer = OptimizerType::SGD,
                    const float& beta1 = 0.9f, const float& beta2 = 0.999f, const float& eps = 1e-8f, const float& weightDecay = 0.01f)
            {
                size_t numSamples = trainingData.size();
                if (numSamples == 0 || batchSize == 0) return;

                float currentLearningRate = learningRate;
                float decayFactor = 0.95f;
                int totalBatches = static_cast<int>((numSamples + batchSize - 1) / batchSize);

                for (int epoch = 0; epoch < epochs; ++epoch)
                {
                    float totalEpochLoss = 0.0f;
                    size_t processedSamples = 0;

                    for (size_t i = 0; i < numSamples; i += batchSize)
                    {
                        size_t currentBatchSize = 0;

                        for (size_t b = 0; b < batchSize && (i + b) < numSamples; b++)
                        {
                            size_t sampleIdx = i + b;
                            const auto& sampleItems = trainingData[sampleIdx];

                            const auto& output = ForwardPass(sampleItems.first);
                            BackPropagation(sampleItems.second);

                            float sampleLoss = 0.0f;
                            for (size_t k = 0; k < output.size(); ++k)
                            {
                                float diff = sampleItems.second[k] - output[k];
                                sampleLoss += diff * diff;
                            }
                            totalEpochLoss += (0.5f * sampleLoss);

                            currentBatchSize++;
                            processedSamples++;
                        }

                        switch (optimizer)
                        {
                            case OptimizerType::AdamW:
                                ApplyBatchAdamW(currentLearningRate, currentBatchSize, beta1, beta2, eps, weightDecay);
                                break;
                            case OptimizerType::SGD:
                                ApplyBatch(currentLearningRate, currentBatchSize);
                                break;
                        }

                        if (onProgress)
                        {
                            int currentBatch = static_cast<int>(i / batchSize) + 1;
                            float averageLoss = totalEpochLoss / processedSamples;
                            onProgress(epoch + 1, epochs, currentBatch, totalBatches, static_cast<int>(processedSamples), static_cast<int>(numSamples), averageLoss);
                        }
                    }

                    if (optimizer == OptimizerType::SGD)
                    {
                        currentLearningRate *= decayFactor;
                    }
                }
            }

            /// @brief Serializes complete network configuration and weights to JSON structure
            /// @return JSON representation containing all underlying network parameter layers
            json CreateJsonFile() const
            {
                size_t layersSize = layers.size();
                vector<json> jsonLayers(layersSize);

                for (size_t i = 0; i < layersSize; i++)
                {
                    jsonLayers[i] = layers[i].CreateJsonLayer();
                }
                return {
                    {"Layers", jsonLayers}
                };
            }

            /// @brief Updates active activation function at target index layer
            /// @param layer Index position target layer
            /// @param activType Desired activation strategy Enum
            /// @param actvFunc Custom forward function callback (optional)
            /// @param invActvFunc Custom derivative function callback (optional)
            void ChangeLayerActivationFunc(const size_t& layer, const ActivationType& activType,
                    const function<float(float)>& actvFunc = nullptr,
                    const function<float(float)>& invActvFunc = nullptr)
            {
                layers[layer].ChangeActivationFunction(activType, actvFunc, invActvFunc);
            }

        private:
            /// @brief Array holding layer models
            vector<Layer> layers;
        
            /// @brief Evaluates error differentials and propagates layer gradient state backward
            /// @param idealOutput Target ground truth evaluation array
            void BackPropagation(const vector<float>& idealOutput)
            {
                const auto &finalOutputs = layers.back().lastOutputs;
                const auto &finalSums = layers.back().lastSums;

                vector<float> deltas(finalOutputs.size());
                for (size_t j = 0; j < finalOutputs.size(); j++)
                {
                    float error = finalOutputs[j] - idealOutput[j];
                    deltas[j] = error * layers.back().invActivationFunction(finalSums[j]);
                }

                for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--)
                {
                    deltas = layers[i].AccumulateGradients(deltas);

                    if (i > 0)
                    {
                        const auto &prevSums = layers[i - 1].lastSums;
                        for (size_t k = 0; k < deltas.size(); k++)
                        {
                            deltas[k] *= layers[i - 1].invActivationFunction(prevSums[k]);
                        }
                    }
                }
            }

            /// @brief Executes SGD parameter step across layers
            /// @param learningRate Active step magnitude scalar
            /// @param batchSize Mini-batch scale denominator
            void ApplyBatch(const float& learningRate, const size_t& batchSize)
            {
                for (auto &layer : layers)
                {
                    layer.ApplyGradients(learningRate, batchSize);
                }
            }

            /// @brief Executes AdamW parameter step across layers
            /// @param learningRate Step learning rate
            /// @param batchSize Mini-batch scale denominator
            /// @param beta1 First moment decay exponential term
            /// @param beta2 Second moment decay exponential term
            /// @param eps Division scalar for numerical stabilization
            /// @param weightDecay Regularization weight penalty
            void ApplyBatchAdamW(const float& learningRate, const size_t& batchSize,
                float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weightDecay = 0.01f)
            {
                for (auto &layer : layers)
                {
                    layer.ApplyGradientsAdamW(learningRate, batchSize, beta1, beta2, eps, weightDecay);
                }
            }
    };
};