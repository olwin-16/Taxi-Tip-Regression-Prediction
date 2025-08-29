# Taxi Tip Prediction - A Comparative Study of Decision Tree Regression Using Scikit-Learn and Snap ML

<br>

<img width="840" height="426" alt="taxi-tip" src="https://github.com/user-attachments/assets/3b5b647c-4e19-4b8c-9731-68c34cf0d950" />

<br>

## Project Overview

This project aims to predict taxi tip amounts using real-world data from the **NYC Taxi and Limousine Commission **(TLC). It applies decision tree regression models implemented via both the widely-used **Scikit-Learn** and the high-performance **Snap ML** library from **IBM**.

The objective is to compare the **speed** and **accuracy** of both libraries by training **Decision Tree Regressors** on this large dataset, harnessing Snap ML’s accelerated **CPU/GPU** implementations for **faster** training without **sacrificing** accuracy. This offers practical experience in machine learning modeling and benchmarking on real datasets with regression tasks.

## Dataset Details

- **Source**: [NYC Taxi and Limousine Commission Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Data Used**: TLC Yellow Taxi Trip Records from June 2019
- **Size:** Initially 3,936,004 observations, subset reduced to 100,000 records for efficiency
- **Variables**: 18 columns capturing trip and payment details, including the tip_amount target

## Data Exploration & Preprocessing

- Cleaned data by removing $0 tips (assumed cash tips), trips where tips **exceed** fares, and trips with fare amounts outside a reasonable range.
- Dropped target-related **total_amount** column to avoid data leakage.
- Converted **pickup** and **dropoff datetime** columns to **hour** and **day** features, and calculated trip duration in **minutes**.
- Limited dataset to **1,000,000** rows to **speed up** training and **reduce** memory usage.
- **One-hot encod**ed categorical features** including **VendorID, RatecodeID, payment type, locations, pickup/dropoff hour** and **day**.
- After **preprocessing**, the dataset expanded to **3,751 features**.
- **Normalized** **feature matrix** using **L1 norm**.
- Split data into **training** (70%) and **testing** (30%) sets.

## Tip Amount Distribution

<img width="569" height="413" alt="image" src="https://github.com/user-attachments/assets/dbf16b3f-5630-4525-bbc3-90725a3b88ef" />

<br>
<br>

- **Histogram** analysis showed the majority of tips were between **0** and **5 dollars**, with a **long right tail**.
- Minimum tip was **$0.01**, maximum **$53.00** per trip.
- **90%** of tips were below or equal to **$5.15**, indicating a **highly skewed distribution**.

## Model Development & Training

### Decision Tree Regressor Using Scikit-Learn

- Model configured with **max depth 8** for **balanced complexity**.
- Trained on the training set, completing in **approximately 3.47 seconds**.

### Decision Tree Regressor Using Snap ML

- **Multi-threaded CPU** training configured with **4 threads**.
- Trained **substantially faster**, completing in **approximately 1.79 seconds**, nearly **twice** as fast as **Scikit-Learn**.

## Model Evaluation & Comparison

| Model        | Training Time (s) | Mean Squared Error (MSE) |
|--------------|-------------------|--------------------------|
| Scikit-Learn | 3.47              | 1.621                    |
| Snap ML      | 1.79              | 1.694                    |

- **Snap ML** delivered **~1.94x** training speedup.
- Both models (Scikit-Learn/Snap ML) achieved **very similar accuracy**, with **Scikit-Learn edging slightly lower MSE**.

## Conclusion

The comparative study demonstrates that the IBM Snap ML library accelerates training of decision tree regression models significantly while offering comparable predictive accuracy versus Scikit-Learn. Snap ML's seamless integration and optimized implementations make it a valuable tool for large-scale machine learning workflows, especially with computationally intensive tree-based models.

## Setup & Installation

```bash
git clone https://github.com/your-username/taxi-tip-prediction.git
cd taxi-tip-prediction
```
```bash
pip install -r requirements.txt
```
```bash
python taxi_tip_prediction_decision_tree_comparison.py
```

## requirements.txt

```bash
snapml
scikit-learn
matplotlib
pandas
numpy
```

## License

Dataset sourced from [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and subject to their terms and conditions.
Code is licensed under the [MIT License](LICENSE).

## Contact

Open an issue or contact via [Email](mailto:olwinchristian1626@gmail.com) for any questions or contributions.
