# Plot Section
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats # P values
from sklearn.ensemble import RandomForestRegressor

# from sklearn.feature_selection import f_regression, mutual_info_regression # P values

def plot_histograms_grouped(X, path, bins=30, max_per_figure=10):
    num_features = X.shape[1]
    num_figures = int(np.ceil(num_features / max_per_figure))
    
    columns = X.columns.tolist()
    
    for fig_idx in range(num_figures):
        start = fig_idx * max_per_figure
        end = min(start + max_per_figure, num_features)
        subset_columns = columns[start:end]
        
        n_cols = 2
        n_rows = int(np.ceil(len(subset_columns) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 4))
        axes = axes.flatten()

        for idx, column in enumerate(subset_columns):
            ax = axes[idx]
            ax.hist(X[column].dropna(), bins=bins, color='skyblue', edgecolor='black')
            ax.set_title(column)
            ax.grid(True, linestyle='--', alpha=0.7)

        for extra_idx in range(len(subset_columns), len(axes)):
            fig.delaxes(axes[extra_idx])

        plt.tight_layout()
        plt.savefig(f"{path}_histograms_{fig_idx}.png")
        plt.close(fig) 

def plot_pie_chart(series, path, title="Pie Chart Y Column"):
    counts = series.value_counts()
    labels = counts.index
    sizes = counts.values

    fig = plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')
    plt.savefig(f"{path}_{title.replace(' ', '_')}.png")
    plt.close(fig)

def plot_heatmap(X,Y,path):
    joined_for_heatmap=X.join(Y)
    correlation_full_health = joined_for_heatmap.corr()

    fig, ax = plt.subplots(figsize=(10, 8)) 
    sns.heatmap(
        correlation_full_health,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(50, 500, n=500),
        square=True,
        ax=ax
    )
    plt.savefig(f"{path}_heatmapFeature.png")
    plt.close(fig)


def calculate_p_values(X, Y):
    """
    Calculate p-values for all numerical features against the target variable
    """
    p_values_table = []
    # For each feature, calculate correlation and p-value
    for feature in X.columns:
        # Pearson correlation test
        corr, p_value = stats.pearsonr(X[feature], Y)
        
        # Determine significance
        significant = "Significant" if p_value <= 0.05 else "Not Significant"
        
        p_values_table.append({
            'Feature': feature,
            'P-Value': p_value,
            'Significance': significant,
            'Correlation': corr
        })
    
    return pd.DataFrame(p_values_table)


def format_p_values_table(p_values_df,path):
    """
    Format the p-values table for better readability
    """
    # Sort by p-value (lowest first)
    p_values_df = p_values_df.sort_values('P-Value')
    
    # Format p-values for better display
    p_values_df['P-Value Formatted'] = p_values_df['P-Value'].apply(
        lambda x: f"{x:.6f}" if x > 0.0001 else f"{x:.2e}"
    )
    
    # Create a styled table
    styled_df = p_values_df[['Feature', 'P-Value Formatted', 'Significance']].copy()
    styled_df.columns = ['Feature', 'P-Value', 'Significance']
    
    # Add color coding
    def color_significant(val):
        color = 'red' if val == 'Not Significant' else 'green'
        return f'color: {color}'
    
    styled_table = styled_df.style.applymap(color_significant, subset=['Significance'])

    results = p_values_df

    with open(f"{path}_p_values.txt", 'w') as f:
        f.write("P-VALUES ANALYSIS\n")
        f.write("================\n\n")
        
        for index, row in results.iterrows():
            p_str = f"{row['P-Value']:.2e}" if row['P-Value'] < 0.001 else f"{row['P-Value']:.6f}"
            f.write(f"{row['Feature']}: {p_str} ({row['Significance']})\n")
        
        # Count significant features
        sig_count = sum(results['Significance'] == 'Significant')
        f.write(f"\nSummary: {sig_count}/{len(results)} features are significant\n")
        
    return styled_table



import shap
from sklearn.model_selection import train_test_split

def SHAP(X,Y,path):
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test,check_additivity=False)
    

    # Create sorted bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Highest to Lowest)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{path}_shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix_roc(fpr, tpr, cm, path):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                cbar=False, annot_kws={'size': 14})
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    roc_auc = np.trapz(tpr, fpr)  
    ax2.plot(fpr, tpr, color='darkorange', lw=3, label=f'AUC = {roc_auc:.3f}')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower right", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{path}_Confusion_Matrix_and_Roc_Curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    