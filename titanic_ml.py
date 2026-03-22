"""
Titanic Survival Predictor - Full ML Pipeline
Skills: Pandas, EDA, Data Cleaning, Scikit-learn, Classification
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, json, os
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. SYNTHETIC TITANIC DATA (realistic stats)
# ─────────────────────────────────────────────
def generate_titanic_data(n=891):
    """Generate realistic Titanic-like data based on historical statistics."""
    # Historical survival rates: 1st class ~63%, 2nd ~47%, 3rd ~24%
    # Women ~74%, Men ~19%; Children higher

    records = []
    passenger_id = 1

    # Class distribution: ~24% 1st, ~21% 2nd, ~55% 3rd
    class_dist = [1]*216 + [2]*184 + [3]*491
    np.random.shuffle(class_dist)

    for pclass in class_dist:
        sex = np.random.choice(['male', 'female'], p=[0.65, 0.35])

        if pclass == 1:
            age_mean, age_std = 39, 14
        elif pclass == 2:
            age_mean, age_std = 29, 12
        else:
            age_mean, age_std = 25, 14

        age = max(1, np.random.normal(age_mean, age_std))
        age = None if np.random.random() < 0.20 else round(age, 1)

        # Survival probability by class + sex + age
        base = {(1,'female'):0.97,(1,'male'):0.37,
                (2,'female'):0.92,(2,'male'):0.16,
                (3,'female'):0.50,(3,'male'):0.15}[pclass, sex]

        # Children boost survival
        eff_age = age if age else age_mean
        if eff_age < 10: base = min(base + 0.25, 0.98)

        survived = int(np.random.random() < base)

        sibsp = np.random.choice([0,1,2,3,4,5,8], p=[0.68,0.17,0.08,0.04,0.01,0.01,0.01])
        parch = np.random.choice([0,1,2,3,4,5,6], p=[0.76,0.13,0.08,0.01,0.01,0.005,0.005])

        fare_base = {1: 84, 2: 20, 3: 13}[pclass]
        fare = max(5, np.random.lognormal(np.log(fare_base), 0.5))
        fare = None if np.random.random() < 0.001 else round(fare, 4)

        embarked = np.random.choice(['S','C','Q'],
                                    p={1:[0.46,0.46,0.08],
                                       2:[0.72,0.18,0.10],
                                       3:[0.72,0.18,0.10]}[pclass])
        embarked = None if np.random.random() < 0.002 else embarked

        title_map = {
            'male':   np.random.choice(['Mr','Master','Dr','Rev'],
                                       p=[0.89,0.06,0.03,0.02]),
            'female': np.random.choice(['Miss','Mrs','Ms','Dr'],
                                       p=[0.48,0.48,0.02,0.02])
        }
        title = title_map[sex]

        records.append({
            'PassengerId': passenger_id,
            'Survived': survived,
            'Pclass': pclass,
            'Name': f'Passenger, {title}. Sample',
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked,
            'Title': title
        })
        passenger_id += 1

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()

    # Title extraction (already in synthetic data)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Age imputation with median by Pclass+Sex
    df['Age'] = df.groupby(['Pclass','Sex'])['Age'].transform(
        lambda x: x.fillna(x.median()))
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Fare imputation
    df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'), inplace=True)

    # Embarked imputation
    df['Embarked'].fillna('S', inplace=True)

    # Age bins
    df['AgeBin'] = pd.cut(df['Age'],
                          bins=[0,12,18,35,60,120],
                          labels=['Child','Teen','Adult','MiddleAge','Senior'])

    # Fare bins
    df['FareBin'] = pd.qcut(df['Fare'], q=4,
                             labels=['Low','Mid','High','VeryHigh'])

    # Encode
    df['Sex_enc']      = (df['Sex'] == 'female').astype(int)
    df['Embarked_enc'] = df['Embarked'].map({'S':0,'C':1,'Q':2})

    title_enc = {'Mr':0,'Mrs':1,'Miss':2,'Master':3,'Dr':4,'Rev':5,'Ms':2}
    df['Title_enc'] = df['Title'].map(title_enc).fillna(0)

    df['AgeBin_enc']  = df['AgeBin'].cat.codes.replace(-1, 0)
    df['FareBin_enc'] = df['FareBin'].cat.codes.replace(-1, 0)

    df['Age_scaled']  = (df['Age'] - df['Age'].mean()) / df['Age'].std()
    df['Fare_scaled'] = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()

    # Final safety fillna
    for col in ['Fare_scaled','Embarked_enc','AgeBin_enc','FareBin_enc']:
        df[col].fillna(0, inplace=True)

    return df


# ─────────────────────────────────────────────
# 3. EDA VISUALIZATIONS
# ─────────────────────────────────────────────
def create_eda_plots(df):
    os.makedirs('/home/claude/plots', exist_ok=True)

    palette = {'survived': '#2ecc71', 'died': '#e74c3c'}
    bg = '#0f1117'
    text_color = '#ecf0f1'

    plt.rcParams.update({
        'figure.facecolor': bg, 'axes.facecolor': '#1a1d2e',
        'axes.edgecolor': '#2c3e50', 'text.color': text_color,
        'axes.labelcolor': text_color, 'xtick.color': text_color,
        'ytick.color': text_color, 'grid.color': '#2c3e50',
        'font.family': 'DejaVu Sans'
    })

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Titanic — Exploratory Data Analysis', fontsize=22,
                 fontweight='bold', color='#f39c12', y=1.01)

    # 1. Survival rate overall
    ax = axes[0, 0]
    counts = df['Survived'].value_counts()
    bars = ax.bar(['Died', 'Survived'], counts.values,
                  color=['#e74c3c', '#2ecc71'], edgecolor='none', width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=11,
                color=text_color, fontweight='bold')
    ax.set_title('Overall Survival', fontsize=14, fontweight='bold', color='#f39c12')
    ax.set_ylabel('Passengers')
    ax.set_ylim(0, counts.max() * 1.2)

    # 2. Survival by Class
    ax = axes[0, 1]
    class_surv = df.groupby('Pclass')['Survived'].mean() * 100
    bars = ax.bar([f'Class {c}' for c in class_surv.index], class_surv.values,
                  color=['#3498db','#9b59b6','#e67e22'], edgecolor='none', width=0.5)
    for bar, val in zip(bars, class_surv.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11, color=text_color, fontweight='bold')
    ax.set_title('Survival Rate by Class', fontsize=14, fontweight='bold', color='#f39c12')
    ax.set_ylabel('Survival Rate (%)')
    ax.set_ylim(0, 100)
    ax.axhline(50, color='#7f8c8d', linestyle='--', alpha=0.5)

    # 3. Survival by Sex
    ax = axes[0, 2]
    sex_surv = df.groupby('Sex')['Survived'].mean() * 100
    bars = ax.bar(['Male', 'Female'], sex_surv.values,
                  color=['#3498db','#e91e8c'], edgecolor='none', width=0.5)
    for bar, val in zip(bars, sex_surv.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11, color=text_color, fontweight='bold')
    ax.set_title('Survival Rate by Sex', fontsize=14, fontweight='bold', color='#f39c12')
    ax.set_ylabel('Survival Rate (%)')
    ax.set_ylim(0, 100)
    ax.axhline(50, color='#7f8c8d', linestyle='--', alpha=0.5)

    # 4. Age Distribution
    ax = axes[1, 0]
    survived = df[df['Survived'] == 1]['Age'].dropna()
    died      = df[df['Survived'] == 0]['Age'].dropna()
    ax.hist(died,      bins=30, alpha=0.7, color='#e74c3c', label='Died',     edgecolor='none')
    ax.hist(survived,  bins=30, alpha=0.7, color='#2ecc71', label='Survived', edgecolor='none')
    ax.set_title('Age Distribution by Survival', fontsize=14, fontweight='bold', color='#f39c12')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    ax.legend(facecolor='#1a1d2e', edgecolor='#2c3e50')

    # 5. Survival by Family Size
    ax = axes[1, 1]
    fam_surv = df.groupby('FamilySize')['Survived'].mean() * 100
    ax.plot(fam_surv.index, fam_surv.values, 'o-', color='#f39c12',
            linewidth=2.5, markersize=8, markerfacecolor='#e74c3c')
    ax.fill_between(fam_surv.index, fam_surv.values, alpha=0.2, color='#f39c12')
    ax.set_title('Survival Rate by Family Size', fontsize=14, fontweight='bold', color='#f39c12')
    ax.set_xlabel('Family Size')
    ax.set_ylabel('Survival Rate (%)')
    ax.set_ylim(0, 100)
    ax.axhline(50, color='#7f8c8d', linestyle='--', alpha=0.5)

    # 6. Fare Distribution
    ax = axes[1, 2]
    fare_bins = ['Low', 'Mid', 'High', 'VeryHigh']
    fare_surv = df.groupby('FareBin', observed=True)['Survived'].mean() * 100
    bars = ax.bar(fare_surv.index.astype(str), fare_surv.values,
                  color=['#1abc9c','#3498db','#9b59b6','#f39c12'],
                  edgecolor='none', width=0.6)
    for bar, val in zip(bars, fare_surv.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11, color=text_color, fontweight='bold')
    ax.set_title('Survival Rate by Fare Tier', fontsize=14, fontweight='bold', color='#f39c12')
    ax.set_ylabel('Survival Rate (%)')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('/home/claude/plots/eda.png', dpi=150, bbox_inches='tight',
                facecolor=bg)
    plt.close()
    print("✓ EDA plots saved")


# ─────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
FEATURES = ['Pclass','Sex_enc','Age_scaled','Fare_scaled',
            'SibSp','Parch','FamilySize','IsAlone',
            'Embarked_enc','Title_enc','AgeBin_enc','FareBin_enc']

def train_models(df):
    X = df[FEATURES].fillna(0)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=6,
                                                       min_samples_split=4, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                           max_depth=4, random_state=42),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_model, best_score = None, 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]

        acc   = accuracy_score(y_test, y_pred)
        auc   = roc_auc_score(y_test, y_prob)
        cv_sc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        results[name] = {
            'model': model,
            'accuracy': acc,
            'auc': auc,
            'cv_mean': cv_sc.mean(),
            'cv_std': cv_sc.std(),
            'y_pred': y_pred,
            'y_prob': y_prob,
            'y_test': y_test
        }

        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  CV Score : {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Died','Survived'])}")

        if acc > best_score:
            best_score = acc
            best_model = name

    print(f"\n🏆  Best Model: {best_model} ({best_score*100:.2f}% accuracy)")
    return results, X_test, y_test, best_model


def plot_model_results(results):
    bg = '#0f1117'
    text_color = '#ecf0f1'
    plt.rcParams.update({
        'figure.facecolor': bg, 'axes.facecolor': '#1a1d2e',
        'axes.edgecolor': '#2c3e50', 'text.color': text_color,
        'axes.labelcolor': text_color, 'xtick.color': text_color,
        'ytick.color': text_color, 'grid.color': '#2c3e50',
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Comparison & Evaluation', fontsize=20, fontweight='bold',
                 color='#f39c12', y=1.02)

    names = list(results.keys())
    short = ['LR', 'RF', 'GB']
    accs  = [results[n]['accuracy'] for n in names]
    aucs  = [results[n]['auc']      for n in names]
    cvs   = [results[n]['cv_mean']  for n in names]
    stds  = [results[n]['cv_std']   for n in names]
    colors = ['#3498db','#2ecc71','#e74c3c']

    # 1. Accuracy comparison
    ax = axes[0]
    bars = ax.bar(short, [a*100 for a in accs], color=colors, edgecolor='none', width=0.5)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val*100:.1f}%', ha='center', fontsize=12, color=text_color, fontweight='bold')
    ax.set_title('Test Accuracy', fontsize=14, fontweight='bold', color='#f39c12')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(70, 100)

    # 2. CV + AUC
    ax = axes[1]
    x = np.arange(len(short))
    w = 0.35
    b1 = ax.bar(x - w/2, [c*100 for c in cvs],  w, color='#9b59b6',
                edgecolor='none', label='CV Accuracy', yerr=[s*100 for s in stds],
                capsize=5, error_kw={'color': text_color})
    b2 = ax.bar(x + w/2, [a*100 for a in aucs],  w, color='#f39c12',
                edgecolor='none', label='ROC-AUC')
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_title('CV Accuracy vs ROC-AUC', fontsize=14, fontweight='bold', color='#f39c12')
    ax.set_ylabel('Score (%)')
    ax.set_ylim(70, 100)
    ax.legend(facecolor='#1a1d2e', edgecolor='#2c3e50')

    # 3. Feature importance (Random Forest)
    ax = axes[2]
    rf = results['Random Forest']['model']
    feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
    feat_labels = {'Pclass':'Class','Sex_enc':'Sex','Age_scaled':'Age',
                   'Fare_scaled':'Fare','SibSp':'Siblings','Parch':'Parents/Children',
                   'FamilySize':'Family Size','IsAlone':'Alone','Embarked_enc':'Embarked',
                   'Title_enc':'Title','AgeBin_enc':'Age Bin','FareBin_enc':'Fare Bin'}
    labels = [feat_labels.get(f, f) for f in feat_imp.index]
    bars = ax.barh(labels, feat_imp.values * 100, color='#3498db', edgecolor='none')
    ax.set_title('Feature Importance (RF)', fontsize=14, fontweight='bold', color='#f39c12')
    ax.set_xlabel('Importance (%)')

    plt.tight_layout()
    plt.savefig('/home/claude/plots/models.png', dpi=150, bbox_inches='tight', facecolor=bg)
    plt.close()
    print("✓ Model plots saved")


# ─────────────────────────────────────────────
# 5. EXPORT MODEL PARAMS FOR FRONTEND
# ─────────────────────────────────────────────
def export_model_params(results, df):
    """Export RF feature importances + stats for interactive frontend."""
    rf = results['Random Forest']['model']
    lr = results['Logistic Regression']['model']

    # Survival stats for UI
    stats = {
        'overall_survival': float(df['Survived'].mean()),
        'class_survival': df.groupby('Pclass')['Survived'].mean().to_dict(),
        'sex_survival':   df.groupby('Sex')['Survived'].mean().to_dict(),
        'model_accuracy': {n: round(results[n]['accuracy']*100,2) for n in results},
        'feature_importance': dict(zip(FEATURES,
                                       [round(float(f)*100,2)
                                        for f in rf.feature_importances_])),
        'age_mean': float(df['Age'].mean()),
        'age_std':  float(df['Age'].std()),
        'fare_mean': float(df['Fare'].mean()),
        'fare_std':  float(df['Fare'].std()),
        'lr_coefs': dict(zip(FEATURES,
                             [round(float(c),4) for c in lr.coef_[0]])),
        'lr_intercept': float(lr.intercept_[0])
    }

    with open('/home/claude/model_params.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("✓ Model params exported to model_params.json")
    return stats


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("   TITANIC SURVIVAL PREDICTOR — ML PIPELINE")
    print("="*55)

    print("\n[1/5] Generating Titanic dataset...")
    df_raw = generate_titanic_data()
    print(f"      Shape: {df_raw.shape}")
    print(f"      Survival rate: {df_raw['Survived'].mean()*100:.1f}%")
    print(f"      Missing Age: {df_raw['Age'].isna().sum()} rows")

    print("\n[2/5] Engineering features...")
    df = engineer_features(df_raw)
    print(f"      Features: {FEATURES}")

    print("\n[3/5] Creating EDA visualizations...")
    create_eda_plots(df)

    print("\n[4/5] Training & evaluating models...")
    results, X_test, y_test, best_model = train_models(df)

    print("\n[5/5] Saving plots & model params...")
    plot_model_results(results)
    stats = export_model_params(results, df)

    print("\n" + "="*55)
    print("   ✅  PIPELINE COMPLETE")
    print("="*55)
    print(f"   Best Model : {best_model}")
    print(f"   Accuracy   : {stats['model_accuracy'][best_model]}%")
    print(f"   Files      : plots/eda.png, plots/models.png, model_params.json")
