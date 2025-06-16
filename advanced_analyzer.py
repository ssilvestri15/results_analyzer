import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLCSAnalysis:
    def __init__(self, analyzer):
        """
        analyzer: istanza di MLCodeSmellCorrelationAnalyzer già processata
        """
        self.analyzer = analyzer
        self.lag_analysis_results = {}
        self.cross_project_results = {}
        
    def lag_correlation_analysis(self, project_name, max_lag=3):
        """
        Analizza correlazioni con ritardo temporale
        Utile per capire se gli effetti si manifestano dopo qualche periodo
        """
        data = self.analyzer.aggregated_data[project_name].copy()
        
        print(f"\n=== LAG CORRELATION ANALYSIS - {project_name} ===")
        
        # Variabili di interesse
        smell_vars = ['smell_density_mean', 'total_smells_found_sum']
        target_vars = ['complexity_delta', 'change_intensity', 'bugfix_ratio']
        
        lag_results = {}
        
        for smell_var in smell_vars:
            for target_var in target_vars:
                if smell_var in data.columns and target_var in data.columns:
                    
                    print(f"\n{smell_var} -> {target_var}")
                    lag_correlations = []
                    
                    for lag in range(max_lag + 1):
                        if lag == 0:
                            # Correlazione contemporanea
                            x = data[smell_var].dropna()
                            y = data[target_var].dropna()
                            
                            # Allinea le serie
                            min_len = min(len(x), len(y))
                            if min_len > 3:
                                r, p = spearmanr(x[:min_len], y[:min_len])
                                lag_correlations.append((lag, r, p, min_len))
                                print(f"  Lag {lag}: r={r:.3f}, p={p:.3f}")
                        else:
                            # Correlazione con ritardo
                            x = data[smell_var].iloc[:-lag].dropna()
                            y = data[target_var].iloc[lag:].dropna()
                            
                            min_len = min(len(x), len(y))
                            if min_len > 3:
                                # Riallinea gli indici
                                x_aligned = x.iloc[:min_len].values
                                y_aligned = y.iloc[:min_len].values
                                
                                r, p = spearmanr(x_aligned, y_aligned)
                                lag_correlations.append((lag, r, p, min_len))
                                print(f"  Lag {lag}: r={r:.3f}, p={p:.3f}")
                    
                    lag_results[f"{smell_var}_to_{target_var}"] = lag_correlations
        
        self.lag_analysis_results[project_name] = lag_results
        return lag_results
    
    def cross_project_meta_analysis(self):
        """
        Analisi meta-analisi cross-project
        Combina i risultati di tutti i progetti per trovare pattern generali
        """
        print("\n=== CROSS-PROJECT META-ANALYSIS ===")
        
        all_correlations = []
        
        # Raccoglie tutte le correlazioni da tutti i progetti
        for project_name, results in self.analyzer.correlation_results.items():
            for category in ['complexity', 'changes', 'bugfixes']:
                if category in results:
                    for test_name, test_result in results[category].items():
                        all_correlations.append({
                            'project': project_name,
                            'category': category,
                            'test': test_name,
                            'correlation': test_result['correlation'],
                            'p_value': test_result['p_value'],
                            'n': test_result['n'],
                            'significant': test_result['significant']
                        })
        
        df_all = pd.DataFrame(all_correlations)
        
        if df_all.empty:
            print("No correlation data available for meta-analysis")
            return None
        
        # Analisi per categoria
        print("\nMETA-ANALYSIS RESULTS:")
        print("-" * 40)
        
        for category in ['complexity', 'changes', 'bugfixes']:
            cat_data = df_all[df_all['category'] == category]
            if not cat_data.empty:
                print(f"\n{category.upper()}:")
                print(f"  Number of tests: {len(cat_data)}")
                print(f"  Significant results: {sum(cat_data['significant'])} ({sum(cat_data['significant'])/len(cat_data)*100:.1f}%)")
                print(f"  Mean correlation: {cat_data['correlation'].mean():.3f}")
                print(f"  Median correlation: {cat_data['correlation'].median():.3f}")
                print(f"  Std correlation: {cat_data['correlation'].std():.3f}")
                
                # Test di consistenza cross-project
                significant_corrs = cat_data[cat_data['significant']]['correlation']
                if len(significant_corrs) > 1:
                    consistency = (significant_corrs > 0).sum() / len(significant_corrs)
                    print(f"  Direction consistency: {consistency:.2f} ({'positive' if consistency > 0.5 else 'negative' if consistency < 0.5 else 'mixed'})")
        
        self.cross_project_results = df_all
        return df_all
    
    def smell_type_specific_analysis(self):
        """
        Analizza correlazioni specifiche per tipo di smell
        Utile se hai diversi tipi di ML code smells
        """
        print("\n=== SMELL TYPE-SPECIFIC ANALYSIS ===")
        
        for project_name in self.analyzer.projects_data.keys():
            commit_data = self.analyzer.projects_data[project_name]['commit_metrics']
            evolution_data = self.analyzer.projects_data[project_name]['smell_evolution']
            
            print(f"\nProject: {project_name}")
            
            # Analizza i tipi di smell presenti
            smell_types = evolution_data.get('smells_by_type', {})
            
            if smell_types:
                print("Smell types found:")
                for smell_type, counts in smell_types.items():
                    total = counts.get('total', 0)
                    active = counts.get('active', 0)
                    print(f"  {smell_type}: {total} total, {active} active")
                
                # Per ogni tipo di smell, analizza le correlazioni
                df = pd.DataFrame(commit_data)
                df['date'] = pd.to_datetime(df['date'])
                
                # Crea aggregazione per tipo di smell se disponibile
                # (questo richiederebbe una struttura dati più dettagliata)
                # Per ora stampiamo solo un summary
                print(f"  -> Detailed analysis would require smell-type specific metrics per commit")
    
    def temporal_stability_analysis(self):
        """
        Analizza la stabilità temporale delle correlazioni
        Divide il dataset in periodi e verifica se le correlazioni rimangono stabili
        """
        print("\n=== TEMPORAL STABILITY ANALYSIS ===")
        
        for project_name, data in self.analyzer.aggregated_data.items():
            if len(data) < 6:  # Serve almeno 6 periodi per fare l'analisi
                continue
                
            print(f"\nProject: {project_name}")
            
            # Dividi in due metà temporali
            mid_point = len(data) // 2
            first_half = data.iloc[:mid_point]
            second_half = data.iloc[mid_point:]
            
            # Test chiave correlazioni per stabilità
            test_pairs = [
                ('smell_density_mean', 'complexity_delta'),
                ('total_smells_found_sum', 'change_intensity'),
                ('smell_density_mean', 'bugfix_ratio')
            ]
            
            for x_var, y_var in test_pairs:
                if x_var in data.columns and y_var in data.columns:
                    
                    # Prima metà
                    if len(first_half) > 3:
                        x1 = first_half[x_var].dropna()
                        y1 = first_half[y_var].dropna()
                        if len(x1) > 2 and len(y1) > 2:
                            min_len1 = min(len(x1), len(y1))
                            r1, p1 = spearmanr(x1[:min_len1], y1[:min_len1])
                        else:
                            r1, p1 = np.nan, np.nan
                    else:
                        r1, p1 = np.nan, np.nan
                    
                    # Seconda metà
                    if len(second_half) > 3:
                        x2 = second_half[x_var].dropna()
                        y2 = second_half[y_var].dropna()
                        if len(x2) > 2 and len(y2) > 2:
                            min_len2 = min(len(x2), len(y2))
                            r2, p2 = spearmanr(x2[:min_len2], y2[:min_len2])
                        else:
                            r2, p2 = np.nan, np.nan
                    else:
                        r2, p2 = np.nan, np.nan
                    
                    if not (np.isnan(r1) or np.isnan(r2)):
                        stability = abs(r1 - r2)
                        print(f"  {x_var} vs {y_var}:")
                        print(f"    First half: r={r1:.3f}, p={p1:.3f}")
                        print(f"    Second half: r={r2:.3f}, p={p2:.3f}")
                        print(f"    Stability (diff): {stability:.3f} {'STABLE' if stability < 0.3 else 'UNSTABLE'}")
    
    def create_advanced_visualizations(self, export_dir=""):
        """
        Crea visualizzazioni avanzate per l'analisi
        """
        # 1. Heatmap delle correlazioni cross-project
        if hasattr(self, 'cross_project_results') and not self.cross_project_results.empty:
            
            # Pivot per heatmap
            pivot_data = self.cross_project_results.pivot_table(
                index='project', 
                columns='category', 
                values='correlation', 
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0, 
                       fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
            plt.title('Cross-Project Correlation Heatmap\n(ML Code Smells vs Quality Metrics)')
            plt.tight_layout()
            if export_dir:
                plt.savefig(f"{export_dir}/cross_project_heatmap.png", dpi=300, bbox_inches='tight')
            else:
                # Salva nella directory corrente
                plt.savefig('cross_project_heatmap.png', dpi=300, bbox_inches='tight')
            #plt.show()
        
        # 2. Forest plot delle correlazioni
        if hasattr(self, 'cross_project_results') and not self.cross_project_results.empty:
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            categories = ['complexity', 'changes', 'bugfixes']
            
            for i, category in enumerate(categories):
                cat_data = self.cross_project_results[
                    self.cross_project_results['category'] == category
                ]
                
                if not cat_data.empty:
                    y_pos = range(len(cat_data))
                    correlations = cat_data['correlation']
                    
                    # Color by significance
                    colors = ['red' if sig else 'gray' for sig in cat_data['significant']]
                    
                    axes[i].scatter(correlations, y_pos, c=colors, alpha=0.7)
                    axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.5)
                    axes[i].set_xlabel('Correlation Coefficient')
                    axes[i].set_ylabel('Study')
                    axes[i].set_title(f'{category.title()} Correlations')
                    axes[i].set_yticks(y_pos)
                    axes[i].set_yticklabels([f"{row['project'][:8]}..." for _, row in cat_data.iterrows()])
            
            plt.tight_layout()
            if export_dir:
                plt.savefig(f"{export_dir}/forest_plot_correlations.png", dpi=300, bbox_inches='tight')
            else:
                # Salva nella directory corrente
                plt.savefig('forest_plot_correlations.png', dpi=300, bbox_inches='tight')
            #plt.show()
    
    def export_advanced_results(self, filename_prefix="advanced_analysis"):
        """
        Esporta i risultati delle analisi avanzate
        """
        # Export lag analysis
        if self.lag_analysis_results:
            with open(f"{filename_prefix}_lag_analysis.json", 'w') as f:
                import json
                json.dump(self.lag_analysis_results, f, indent=2, default=str)
        
        # Export cross-project results
        if hasattr(self, 'cross_project_results') and not self.cross_project_results.empty:
            self.cross_project_results.to_csv(f"{filename_prefix}_cross_project.csv", index=False)
        
        print(f"Advanced analysis results exported with prefix: {filename_prefix}")

# Esempio di utilizzo delle analisi avanzate
def run_advanced_analysis():
    """
    Esegui le analisi avanzate dopo l'analisi base
    """
    # Assumendo che hai già eseguito l'analisi base
    from main_script import analyzer  # Importa l'analyzer già processato
    
    # Crea l'analyzer avanzato
    advanced_analyzer = AdvancedMLCSAnalysis(analyzer)
    
    # Esegui analisi con lag temporale
    for project in analyzer.projects_data.keys():
        advanced_analyzer.lag_correlation_analysis(project, max_lag=3)
    
    # Meta-analisi cross-project
    advanced_analyzer.cross_project_meta_analysis()
    
    # Analisi specifica per tipo di smell
    advanced_analyzer.smell_type_specific_analysis()
    
    # Analisi stabilità temporale
    advanced_analyzer.temporal_stability_analysis()
    
    # Esporta risultati
    advanced_analyzer.export_advanced_results()

if __name__ == "__main__":
    run_advanced_analysis()