# stock-market-prediction using Deep Learning (GANs)-sentiment-analysis

This project aims to predict stock prices in the Toronto Stock Market using a unique deep learning model that utilizes generative adversarial networks (GANs). The GAN model is trained on 22 years of true historical stock data from the Toronto Stock Exchange. The objective is to provide recommendations on which stocks to buy over a specific period of time.

## Methodology

The methodology employed in this project involves the following steps:

1. Data Collection: Gather 22 years of true historical stock data from the Toronto Stock Exchange.

2. GAN Model Training: Utilize generative adversarial networks (GANs) to train the deep learning model. GANs consist of two neural networks - a generator network that generates synthetic data and a discriminator network that distinguishes between real and synthetic data. The GAN model is trained to generate realistic stock price predictions.

3. Model Evaluation: Compare the performance of the GAN model against various machine learning and deep learning methods such as Random Forests (RF), gradient boosting (GB), Support Vector Machines (SVM), and long-short term memory (LSTM). Several evaluation metrics, including root mean square error (RMSE), mean squared error (MSE), maximum error (ME), R-squared (R2), explained variance score (EVS), and mean absolute error (MAE), are used to assess the performance of the models.

4. Enhancement with a Temporal Attention Layer: Improve the GAN model by incorporating a Temporal Attention Layer. This layer aids in identifying and forecasting market movements, enhancing the accuracy of the predictions.

5. Incorporation of External Data: Enhance the accuracy of projections by incorporating data from news and social networking sites. This additional data can provide valuable insights and help improve the forecasting capabilities of the model.

## Repository Structure

- `data/`: This directory contains the historical stock data from the Toronto Stock Exchange used for training and evaluation.

- `models/`: This directory contains the trained GAN model and other machine learning models used for comparison.

- `scripts/`: This directory contains the scripts and code for data preprocessing, model training, and evaluation.

- `notebooks/`: This directory contains Jupyter notebooks with detailed explanations and step-by-step implementation of the GAN model and other machine learning models.

- `results/`: This directory contains the evaluation results and performance metrics of the different models.

## Installation

To run this project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/stock-market-prediction.git`

2. Install the required dependencies: `pip install -r requirements.txt`

3. Run the data preprocessing script: `python scripts/preprocess_data.py`

4. Train the GAN model: `python scripts/train_gan.py`

5. Evaluate the model and compare against other methods: `python scripts/evaluate_models.py`

6. Access the Jupyter notebooks in the `notebooks/` directory for a detailed explanation and implementation of the models.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Contact

For any questions or suggestions, feel free to reach out to the project maintainer:

- Name: Inam Ullah 
- Email: Inamullahiiufet@gmail.com

**Note:** This README provides an overview of the project and its structure. For detailed explanations and step-by-step implementation, please refer to the Jupyter notebooks in the `notebooks/` directory.
