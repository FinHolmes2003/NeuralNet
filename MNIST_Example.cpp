#include <wx/wxprec.h>
#ifndef WX_PRECOMP
    #include <wx/wx.h>
#endif

#include "include/Network.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <nlohmann/json.hpp>
#include <filesystem>

using json = nlohmann::json;

enum
{
    ID_DisplayImage = 1,
    ID_TrainAI = 2,
    ID_SaveWeights = 3,
    ID_LoadWeights = 4,
};

class MyFrame : public wxFrame
{
    public:
        /// @brief Constructor for the frame. Builds the menu options and image canvas
        MyFrame(Network::Network network)
            : wxFrame(NULL, wxID_ANY, "Neural Network MNIST test", wxDefaultPosition, wxSize(750, 500))
        {
            neuralNet = network;

            wxMenu *menuFile = new wxMenu; // menu titled "File" at top of window
            menuFile->Append(ID_DisplayImage, "&Next MNIST Digit...\tCtrl-N", "Load the next digit from the dataset");
            menuFile->Append(ID_TrainAI, "&Train Neural Network... \tCtrl-T", "Trains the Neural network on the MNIST dataset");
            menuFile->Append(ID_SaveWeights, "&Save Neural Network... \tCtrl-S", "Save the weights and biases of the neural network to /Weights/MNIST.json");
            menuFile->Append(ID_LoadWeights, "&Load network... \tCtrl-L", "Load the previously saved weights and biases");
            menuFile->AppendSeparator();
            menuFile->Append(wxID_EXIT);

            wxMenuBar *menuBar = new wxMenuBar;
            menuBar->Append(menuFile, "&File");

            SetMenuBar(menuBar);
            CreateStatusBar();
            SetStatusText("Click File -> Next MNIST Digit to view handwritten samples."); // text at bottom of screen

            m_panel = new wxPanel(this);
            wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);

            wxImage placeholderImg(200, 200); // image size
            placeholderImg.SetRGB(wxRect(0, 0, 200, 200), 240, 240, 240); // sets the inital image as a white square
            
            m_bitmapCtrl = new wxStaticBitmap(m_panel, wxID_ANY, wxBitmap(placeholderImg));
            sizer->Add(m_bitmapCtrl, 0, wxALIGN_CENTER | wxALL, 20);

            m_infoTextCtrl = new wxTextCtrl(m_panel, wxID_ANY, "", wxDefaultPosition, wxSize(300, 30), wxTE_CENTER | wxTE_READONLY);
            sizer->Add(m_infoTextCtrl, 0, wxALIGN_CENTER | wxBOTTOM, 10);

            m_panel->SetSizer(sizer); // layout the image in the middle ofthe frame
            sizer->Layout();

            Bind(wxEVT_MENU, &MyFrame::OnDisplayImage, this, ID_DisplayImage); // set all the methods for the menu buttons
            Bind(wxEVT_MENU, &MyFrame::OnExit, this, wxID_EXIT);
            Bind(wxEVT_MENU, &MyFrame::OnTrain, this, ID_TrainAI);
            Bind(wxEVT_MENU, &MyFrame::OnSave, this, ID_SaveWeights);
            Bind(wxEVT_MENU, &MyFrame::OnLoad, this, ID_LoadWeights);
        }

    private:
        Network::Network neuralNet;
        wxPanel* m_panel;
        wxStaticBitmap* m_bitmapCtrl;
        wxTextCtrl* m_infoTextCtrl;
        int m_currentImageIndex = 0;
        static constexpr size_t imageWidth = 28;
        static constexpr size_t imageHeight = 28;

        // Endianness swapper for Big-Endian MNIST binary files
        static uint32_t SwapEndian(uint32_t val)
        {
            return ((val << 24) & 0xFF000000) |
                   ((val << 8)  & 0x00FF0000) |
                   ((val >> 8)  & 0x0000FF00) |
                   ((val >> 24) & 0x000000FF);
        }

        /// @brief loads the images and labels from the MNIST dataset
        /// @param imagePath 
        /// @param labelPath 
        /// @param numSamples if set to 0 it will load all the images in the file
        /// @return array of pairs of the colour value for each pixel and the label as an array of [0-9] where it is only 1.0 on the correct number
        std::vector<std::pair<std::vector<double>, std::vector<double>>> LoadMnistDataset(
            const std::string& imagePath = "MNIST_ORG/train-images.idx3-ubyte", 
            const std::string& labelPath = "MNIST_ORG/train-labels.idx1-ubyte", 
            size_t numSamples = 0)
        {
            std::ifstream imgFile(imagePath, std::ios::binary); // load files
            std::ifstream lblFile(labelPath, std::ios::binary);

            if (!imgFile.is_open() || !lblFile.is_open())
            {
                std::cerr << "Failed to open MNIST files for training conversion.\n";
                return {};
            }

            uint32_t numImages = 0;
            imgFile.seekg(4, std::ios::beg);
            imgFile.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
            numImages = SwapEndian(numImages); // Convert Big-Endian to CPU format

            imgFile.seekg(16, std::ios::beg); // skip header data (16 bytes for images)
            lblFile.seekg(8, std::ios::beg);  // skip header data (8 bytes for labels)

            numSamples = (numSamples == 0 || numSamples > numImages) ? numImages : numSamples;

            constexpr size_t imageSize = imageWidth * imageHeight; // size of mnist images

            std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset;
            dataset.reserve(numSamples);

            std::vector<unsigned char> pixelBuffer(imageSize);
            unsigned char labelByte = 0;

            for (size_t i = 0; i < numSamples; ++i) // for each image needed
            {
                lblFile.read(reinterpret_cast<char*>(&labelByte), 1); // gets the label
                imgFile.read(reinterpret_cast<char*>(pixelBuffer.data()), imageSize); // gets the image

                if (imgFile.gcount() < imageSize || lblFile.gcount() < 1) // if there are none left
                    break;

                std::vector<double> inputValues(imageSize);
                for (size_t p = 0; p < imageSize; ++p)
                {
                    inputValues[p] = static_cast<double>(pixelBuffer[p]) / 255.0; // normalises the pixel
                }

                std::vector<double> targetValues(10, 0.0);
                if (labelByte < 10)
                {
                    targetValues[labelByte] = 1.0; // creates the output vector
                }

                dataset.emplace_back(std::move(inputValues), std::move(targetValues));
            }

            return dataset;
        }

        /// @brief loads a single image sample by jumping directly to its file offset
        /// @param index 
        /// @param outPixels 
        /// @param outLabel 
        /// @param imagePath 
        /// @param labelPath 
        /// @return 
        bool ReadSingleMnistSample(size_t index, std::vector<double>& outPixels, int& outLabel,
                                   const std::string& imagePath = "MNIST_ORG/train-images.idx3-ubyte", 
                                   const std::string& labelPath = "MNIST_ORG/train-labels.idx1-ubyte")
        {
            std::ifstream imgFile(imagePath, std::ios::binary);
            std::ifstream lblFile(labelPath, std::ios::binary);

            if (!imgFile.is_open() || !lblFile.is_open()) return false;

            constexpr size_t imageSize = imageWidth * imageHeight;

            // Jump directly to image byte offset
            imgFile.seekg(16 + index * imageSize, std::ios::beg);
            lblFile.seekg(8 + index, std::ios::beg);

            std::vector<unsigned char> pixelBuffer(imageSize);
            unsigned char labelByte = 0;

            imgFile.read(reinterpret_cast<char*>(pixelBuffer.data()), imageSize);
            lblFile.read(reinterpret_cast<char*>(&labelByte), 1);

            if (imgFile.gcount() < imageSize || lblFile.gcount() < 1) return false;

            outPixels.resize(imageSize);
            for (size_t p = 0; p < imageSize; ++p)
            {
                outPixels[p] = static_cast<double>(pixelBuffer[p]) / 255.0;
            }

            outLabel = static_cast<int>(labelByte);
            return true;
        }

        /// @brief displays the image in the frame
        /// @param normalizedPixels vector or pixels normalized between 0.0 and 1.0
        /// @param width width of the image
        /// @param height height of the image
        void DisplayGrayscaleImage(const std::vector<double>& normalizedPixels, int width, int height)
        {
            unsigned char* rgbData = (unsigned char*)malloc(width * height * 3);

            for (int i = 0; i < width * height; ++i)
            {
                unsigned char val = static_cast<unsigned char>(normalizedPixels[i] * 255.0); // reconverts back to RGB data
                rgbData[i * 3 + 0] = val;
                rgbData[i * 3 + 1] = val;
                rgbData[i * 3 + 2] = val;
            }

            wxImage img(width, height, rgbData, true); // creates the image
            wxImage scaledImg = img.Scale(200, 200, wxIMAGE_QUALITY_NEAREST); // scales it to 200x200

            m_bitmapCtrl->SetBitmap(wxBitmap(scaledImg)); // displays the image
            m_panel->GetSizer()->Layout();
            m_bitmapCtrl->Refresh();
            m_bitmapCtrl->Update();
        }

        /// @brief Saves the weights and biases to the weights folder
        /// @param event 
        void OnSave(wxCommandEvent& event)
        {
            std::filesystem::create_directory("Weights");
            std::ofstream outputFile("Weights/MNIST.json");

            if (outputFile.is_open())
            {
                outputFile << neuralNet.CreateJsonFile().dump(4);
                outputFile.close();
                SetStatusText("Successfully saved network weights.");
            }
        }

        /// @brief Loads the weights and biases from Weights/MNIST.json
        /// @param event 
        void OnLoad(wxCommandEvent& event)
        {
            std::ifstream inputFile("Weights/MNIST.json");
            json loadedFile;
            
            if (inputFile.is_open())
            {
                inputFile >> loadedFile;
                inputFile.close();
                std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> weightsAndBiases = Network::JsonToVectors(loadedFile);
                neuralNet = Network::Network(weightsAndBiases.first, weightsAndBiases.second);
                SetStatusText("Successfully loaded weights and biases.");
            }
            else
            {
                SetStatusText("Failed to load file.");
            }
        }

        /// @brief trains the neural network according to all the images
        /// @param event 
        void OnTrain(wxCommandEvent& event)
        {
            SetStatusText("Loading dataset...");
            wxYield();

            auto testData = LoadMnistDataset();
            if (testData.empty()) return;

            int epochs = 5;
            size_t batchSize = 120;
            double learningRate = 5.0;


            neuralNet.Train(testData, epochs, batchSize, learningRate, 
                [this](int currentEpoch, int totalEpochs, int currentBatch, int totalBatches, int trainingSample, int trainingSampleTotal, double loss) 
                {
                    SetStatusText(wxString::Format("Training... Epoch %d / %d | Batch %d / %d | Image %d / %d | Loss: %.2f", 
                                                currentEpoch, totalEpochs, currentBatch, totalBatches, trainingSample, trainingSampleTotal , loss));
                    wxYield();
                }
            );

            SetStatusText("Training complete!");
            wxMessageBox("Training complete!", "Success", wxOK | wxICON_INFORMATION);
        }

        /// @brief displays the image in the screen
        /// @param event 
        void OnDisplayImage(wxCommandEvent& event)
        {
            static std::random_device rd; // generates random number for the image
            static std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dist(0, 59999);

            m_currentImageIndex = dist(gen);

            std::vector<double> inputPixels;
            int label = -1;

            // Direct binary seek for instantaneous lookup
            if (!ReadSingleMnistSample(m_currentImageIndex, inputPixels, label))
            {
                SetStatusText("Error: Unable to load MNIST sample.");
                return;
            }

            std::vector<double> networkGuess = neuralNet.ForwardPass(inputPixels); // finds the neural network's guess for the image
            int netGuess = std::distance(networkGuess.begin(), std::max_element(networkGuess.begin(), networkGuess.end()));
            double certainty = networkGuess[netGuess] * 100.0;

            DisplayGrayscaleImage(inputPixels, imageWidth, imageHeight);
            SetStatusText(wxString::Format("Loaded MNIST Sample #%d | Label: %d | Network Guess: %d with %.2f%% certainty", 
                                           m_currentImageIndex, label, netGuess, certainty));
            

            std::string success = label == netGuess ? "Success" : "Failure";
            m_infoTextCtrl->SetValue(wxString::Format("%s", success));
        }

        void OnExit(wxCommandEvent& event)
        {
            Close(true);
        }
};

class MyApp : public wxApp
{
    public:
        bool OnInit() override
        {
            std::vector<int> layerSizes{28*28, 100, 100, 10};
            Network::Network neuralNet(layerSizes);

            wxInitAllImageHandlers();
            MyFrame *frame = new MyFrame(neuralNet);
            frame->Show(true);
            return true;
        }
};

wxIMPLEMENT_APP(MyApp);