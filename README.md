# Machine learning in optical networks

* We are using https://github.com/BJTUSensor/Phi-OTDR_dataset_and_codes dataset

optical-network-fault-detection/
│
├── das_data/                 # Directory for datasets
├── neural_network_models/    # Contains model architectures
│   ├── mlp.py                # MLP model definition
│   ├── cnn.py                # CNN model definition
│   └── rnn.py                # RNN model definition
├── train_test.py             # Script for trainingand testing
└── README.md                 # Project README


  ## How to run

  * Firstly downlaoading data from https://drive.google.com/drive/folders/1-4jGDVrGP-KZ-EvLlURN8Y2fJ_IDux-e?usp=sharing
 
  * and start by typing for example ` python train.py --model [mlp|cnn|rnn]`

