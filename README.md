# Insurance Charges Prediction

A beginner machine learning project where I predict how much a person's medical insurance bill will be, based on things like age, BMI, smoking status and region.

I built this to practice the full workflow: exploring data, cleaning it, picking features, training a model and checking how well it works.

## Dataset

`insurance.csv`: 1,338 rows, 7 columns, no missing values.

| Column | What it means |
|---|---|
| age | Age of the person |
| sex | male / female |
| bmi | Body mass index |
| children | Number of children covered |
| smoker | yes / no |
| region | Area in the US (northeast, northwest, southeast, southwest) |
| charges | The insurance bill (this is what I predict) |

## What I did

1. **Explored the data**: histograms for the numeric columns, count plots for sex and region, and bar plots of charges against smoker status, age group, BMI range and number of children.
2. **Cleaned it**: removed 1 duplicate row, converted `sex` and `smoker` to 0/1, one-hot encoded `region`, and created BMI categories (Underweight, Normal, Overweight, Obese Class 1/2/3).
3. **Scaled** `age`, `bmi` and `children` with `StandardScaler`.
4. **Looked at which features matter** using Pearson correlation and chi-square tests.
5. **Trained a Linear Regression model** on an 80/20 train/test split (`random_state=42`).

## Results

| Metric | Score |
|---|---|
| R² | 0.806 |
| Adjusted R² | 0.799 |

So the model explains roughly 80% of the variation in insurance charges on data it had not seen.

## What I found

- **Smoking is by far the biggest factor.** It has a correlation of about 0.79 with charges, much higher than anything else.
- Age comes second (about 0.30), then BMI (about 0.20).
- Sex, region and number of children have only a small effect.

## Things I know I can improve

I'm keeping this list here honestly, because I'd rather show what I've learned than pretend it's perfect.

- The scaler and feature selection were done on the full dataset before splitting. The cleaner way is to split first, then fit them on the training data only.
- A single train/test split can be lucky. Cross-validation would give a more reliable score.
- Linear regression can't capture that smoking and high BMI together push charges up much more than either alone. Adding an interaction feature (smoker × BMI) or trying a tree-based model like Gradient Boosting should help a lot.
- Add RMSE and MAE so the error is shown in real money, not just R².

## How to run it

```bash
git clone <your-repo-link>
cd <your-repo-folder>
pip install -r requirements.txt
jupyter notebook
```

Then open `insurance-cost-prediction` and run all cells. Keep `insurance.csv` in the same folder as the notebook.

## Tools used

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, SciPy
