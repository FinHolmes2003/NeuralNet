#include "include/Network.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>

using json = nlohmann::json;
using namespace std;

int main()
{
    vector<int> layerSetup = {2, 3, 1};

    ifstream inputFile("Weights/weights.json");
    json loadedFile;

    Network::Network neuralnet;

    if (inputFile.is_open()) // if it can load the file with the weights
    {
        inputFile >> loadedFile;
        inputFile.close();
        auto [biases, weights] = Network::JsonToVectors(loadedFile);
        neuralnet = Network::Network(biases, weights); // create the network with the saved weights
    }
    else
    {
        cout << "Could not open weights file. Check path.\n";
        neuralnet = Network::Network(layerSetup); // creates a randomised network if it couldn't load the file
    }

    vector<pair<vector<float>, vector<float>>> trainingData = { // training data to find XOR results
        {{0.0, 0.0}, {0.0}},
        {{0.0, 1.0}, {1.0}},
        {{1.0, 0.0}, {1.0}},
        {{1.0, 1.0}, {0.0}}};

    int epochs = 100000;
    size_t batchSize = trainingData.size();
    float learningRate = 0.5;

    neuralnet.Train(trainingData, epochs, batchSize, learningRate); // trains the data on the values listed above

    for (const auto &sample : trainingData)
    {
        const auto &input = sample.first;
        const auto &target = sample.second;

        cout << input[0] << ", " << input[1] << " -> Output: " 
             << neuralnet.ForwardPass(input)[0] 
             << " (Target: " << target[0] << ")\n";
    }

    cout << "\n" << neuralnet.VisualNetwork() << "\n";

    std::filesystem::create_directory("Weights");
    ofstream outputFile;
    outputFile.open("Weights/weights.json"); // saves the new weights

    if (outputFile.is_open())
    {
        outputFile << neuralnet.CreateJsonFile().dump(4);
        outputFile.close();
    }

    return 0;
}