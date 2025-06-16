import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, shapiro
import matplotlib.pyplot as plt
import matplotlib.figure as Figure
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLCodeSmellCorrelationAnalyzer:
    def __init__(self, projects_data):
        """
        projects_data: dict con chiavi = nomi progetti, valori = dict con commit_metrics, file_frequencies, smell_evolution
        """
        self.projects_data = projects_data
        self.aggregated_data = {}
        self.correlation_results = {}
    
    def load_project_data(self, project_name, commit_file, freq_file, evolution_file):
        """Carica i dati di un singolo progetto"""
        with open(commit_file, 'r') as f:
            commit_data = json.load(f)
        with open(freq_file, 'r') as f:
            freq_data = json.load(f)
        with open(evolution_file, 'r') as f:
            evolution_data = json.load(f)
        
        self.projects_data[project_name] = {
            'commit_metrics': commit_data,
            'file_frequencies': freq_data,
            'smell_evolution': evolution_data
        }
    
    def aggregate_data_by_time(self, project_name, window='M'):
        """
        Aggrega i dati per finestre temporali
        window: 'M'=mese, 'Q'=trimestre, 'W'=settimana
        """
        commit_data = self.projects_data[project_name]['commit_metrics']
        
        # Converti in DataFrame
        df = pd.DataFrame(commit_data)
        df['date'] = pd.to_datetime(df['date'])
        df['period'] = df['date'].dt.to_period(window)
        
        # Aggregazione per periodo
        aggregated = df.groupby('period').agg({
            # ML Code Smells metrics
            'total_smells_found': ['sum', 'mean'],
            'smell_density': 'mean',
            'smells_introduced': 'sum',
            'smells_removed': 'sum',
            
            # Complexity metrics
            'project_cyclomatic_complexity': ['last', 'mean'],
            'commit_cyclomatic_complexity': 'mean',
            
            # Change metrics
            'files_changed': ['sum', 'mean'],
            'LOC_added': 'sum',
            'LOC_deleted': 'sum',
            
            # Bug fix metrics
            'is_bug_fix': 'sum',
            'bug_fixing': 'sum'
        }).reset_index()
        
        # Flattening colonne multi-level
        aggregated.columns = ['_'.join(col).strip() if col[1] else col[0] for col in aggregated.columns.values]
        aggregated.rename(columns={'period_': 'period'}, inplace=True)
        
        # Calcola metriche derivate
        total_commits = df.groupby('period').size().reset_index(name='total_commits')
        aggregated = aggregated.merge(total_commits, on='period')
        
        # Complessità: incremento nel tempo
        aggregated['complexity_delta'] = aggregated['project_cyclomatic_complexity_last'].diff()
        aggregated['complexity_growth_rate'] = aggregated['complexity_delta'] / aggregated['project_cyclomatic_complexity_last'].shift(1)
        
        # Change intensity
        aggregated['change_intensity'] = aggregated['files_changed_sum'] / aggregated['total_commits']
        aggregated['loc_churn'] = (aggregated['LOC_added_sum'] + aggregated['LOC_deleted_sum']) / aggregated['total_commits']
        
        # Bug fix ratio
        aggregated['bugfix_commits'] = aggregated['is_bug_fix_sum'] + aggregated['bug_fixing_sum']
        aggregated['bugfix_ratio'] = aggregated['bugfix_commits'] / aggregated['total_commits']
        
        # ML-CS evolution metrics
        aggregated['smell_introduction_rate'] = aggregated['smells_introduced_sum'] / aggregated['total_commits']
        aggregated['smell_removal_rate'] = aggregated['smells_removed_sum'] / aggregated['total_commits']
        aggregated['net_smell_change'] = aggregated['smells_introduced_sum'] - aggregated['smells_removed_sum']
        
        self.aggregated_data[project_name] = aggregated
        return aggregated
    
    def test_normality(self, data, variable):
        """Test di normalità Shapiro-Wilk"""
        clean_data = data[variable].dropna()
        if len(clean_data) > 3:
            stat, p = shapiro(clean_data)
            return p > 0.05, p
        return False, 1.0
    
    def perform_correlation_analysis(self, project_name):
        """Esegue l'analisi di correlazione per un progetto"""
        data = self.aggregated_data[project_name]
        results = {}
        
        # Domanda 1: ML-CSs vs Complessità del codice
        print(f"\n=== PROGETTO: {project_name} ===")
        print("\n1. CORRELAZIONE: ML Code Smells vs Complessità del Codice")
        
        # Diverse metriche di smell vs complessità
        smell_metrics = ['smell_density_mean', 'total_smells_found_sum', 'smell_introduction_rate']
        complexity_metrics = ['complexity_delta', 'complexity_growth_rate', 'project_cyclomatic_complexity_last']
        
        complexity_results = {}
        for sm in smell_metrics:
            for cm in complexity_metrics:
                if sm in data.columns and cm in data.columns:
                    x = data[sm].dropna()
                    y = data[cm].dropna()
                    
                    # Assicurati che abbiano la stessa lunghezza
                    common_idx = data[sm].notna() & data[cm].notna()
                    x = data.loc[common_idx, sm]
                    y = data.loc[common_idx, cm]
                    
                    if len(x) > 3:
                        # Test normalità
                        x_normal, x_p = self.test_normality(pd.DataFrame({sm: x}), sm)
                        y_normal, y_p = self.test_normality(pd.DataFrame({cm: y}), cm)
                        
                        # Scegli il test appropriato
                        if x_normal and y_normal:
                            r, p = pearsonr(x, y)
                            test_used = "Pearson"
                        else:
                            r, p = spearmanr(x, y)
                            test_used = "Spearman"
                        
                        complexity_results[f"{sm}_vs_{cm}"] = {
                            'correlation': r,
                            'p_value': p,
                            'test': test_used,
                            'n': len(x),
                            'significant': p < 0.05
                        }
                        
                        print(f"  {sm} vs {cm}:")
                        print(f"    {test_used}: r={r:.3f}, p={p:.3f}, n={len(x)} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
        
        results['complexity'] = complexity_results
        
        # Domanda 2: ML-CSs vs Numero di change
        print("\n2. CORRELAZIONE: ML Code Smells vs Attività di Change")
        
        change_metrics = ['change_intensity', 'files_changed_sum', 'loc_churn']
        change_results = {}
        
        for sm in smell_metrics:
            for chm in change_metrics:
                if sm in data.columns and chm in data.columns:
                    common_idx = data[sm].notna() & data[chm].notna()
                    x = data.loc[common_idx, sm]
                    y = data.loc[common_idx, chm]
                    
                    if len(x) > 3:
                        x_normal, _ = self.test_normality(pd.DataFrame({sm: x}), sm)
                        y_normal, _ = self.test_normality(pd.DataFrame({chm: y}), chm)
                        
                        if x_normal and y_normal:
                            r, p = pearsonr(x, y)
                            test_used = "Pearson"
                        else:
                            r, p = spearmanr(x, y)
                            test_used = "Spearman"
                        
                        change_results[f"{sm}_vs_{chm}"] = {
                            'correlation': r,
                            'p_value': p,
                            'test': test_used,
                            'n': len(x),
                            'significant': p < 0.05
                        }
                        
                        print(f"  {sm} vs {chm}:")
                        print(f"    {test_used}: r={r:.3f}, p={p:.3f}, n={len(x)} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
        
        results['changes'] = change_results
        
        # Domanda 3: ML-CSs vs Bug fix activities
        print("\n3. CORRELAZIONE: ML Code Smells vs Bug Fix Activities")
        
        bugfix_metrics = ['bugfix_ratio', 'bugfix_commits']
        bugfix_results = {}
        
        for sm in smell_metrics:
            for bm in bugfix_metrics:
                if sm in data.columns and bm in data.columns:
                    common_idx = data[sm].notna() & data[bm].notna()
                    x = data.loc[common_idx, sm]
                    y = data.loc[common_idx, bm]
                    
                    if len(x) > 3:
                        x_normal, _ = self.test_normality(pd.DataFrame({sm: x}), sm)
                        y_normal, _ = self.test_normality(pd.DataFrame({bm: y}), bm)
                        
                        if x_normal and y_normal:
                            r, p = pearsonr(x, y)
                            test_used = "Pearson"
                        else:
                            r, p = spearmanr(x, y)
                            test_used = "Spearman"
                        
                        bugfix_results[f"{sm}_vs_{bm}"] = {
                            'correlation': r,
                            'p_value': p,
                            'test': test_used,
                            'n': len(x),
                            'significant': p < 0.05
                        }
                        
                        print(f"  {sm} vs {bm}:")
                        print(f"    {test_used}: r={r:.3f}, p={p:.3f}, n={len(x)} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
        
        results['bugfixes'] = bugfix_results
        
        self.correlation_results[project_name] = results
        return results
    
    def create_visualizations(self, export_dir = ""):
        """Crea visualizzazioni per l'analisi"""
        for project_name, data in self.aggregated_data.items():
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Analisi Correlazione ML Code Smells - {project_name}', fontsize=16)

            # Plot 1: Smell density vs Complexity delta
            if 'smell_density_mean' in data.columns and 'complexity_delta' in data.columns:
                axes[0,0].scatter(data['smell_density_mean'], data['complexity_delta'], alpha=0.7)
                axes[0,0].set_xlabel('Smell Density')
                axes[0,0].set_ylabel('Complexity Delta')
                axes[0,0].set_title('Smells vs Complexity Change')

            # Plot 2: Total smells vs Change intensity
            if 'total_smells_found_sum' in data.columns and 'change_intensity' in data.columns:
                axes[0,1].scatter(data['total_smells_found_sum'], data['change_intensity'], alpha=0.7)
                axes[0,1].set_xlabel('Total Smells Found')
                axes[0,1].set_ylabel('Change Intensity')
                axes[0,1].set_title('Smells vs Change Activity')

            # Plot 3: Smell density vs Bug fix ratio
            if 'smell_density_mean' in data.columns and 'bugfix_ratio' in data.columns:
                axes[0,2].scatter(data['smell_density_mean'], data['bugfix_ratio'], alpha=0.7)
                axes[0,2].set_xlabel('Smell Density')
                axes[0,2].set_ylabel('Bug Fix Ratio')
                axes[0,2].set_title('Smells vs Bug Fixes')

            # Time series plots
            data['period_num'] = range(len(data))

            # Plot 4: Evolution of smells over time
            if 'total_smells_found_sum' in data.columns:
                axes[1,0].plot(data['period_num'], data['total_smells_found_sum'], marker='o')
                axes[1,0].set_xlabel('Time Period')
                axes[1,0].set_ylabel('Total Smells')
                axes[1,0].set_title('Smell Evolution Over Time')

            # Plot 5: Evolution of complexity over time
            if 'project_cyclomatic_complexity_last' in data.columns:
                axes[1,1].plot(data['period_num'], data['project_cyclomatic_complexity_last'], marker='o', color='orange')
                axes[1,1].set_xlabel('Time Period')
                axes[1,1].set_ylabel('Project Complexity')
                axes[1,1].set_title('Complexity Evolution Over Time')

            # Plot 6: Bug fix ratio over time
            if 'bugfix_ratio' in data.columns:
                axes[1,2].plot(data['period_num'], data['bugfix_ratio'], marker='o', color='red')
                axes[1,2].set_xlabel('Time Period')
                axes[1,2].set_ylabel('Bug Fix Ratio')
                axes[1,2].set_title('Bug Fix Activity Over Time')

            plt.tight_layout()
            plt.savefig(f'{export_dir}/{project_name}_correlation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    def generate_summary_report(self):
        """Genera un report riassuntivo per tutti i progetti"""
        print("\n" + "="*80)
        print("SUMMARY REPORT - ML CODE SMELLS CORRELATION ANALYSIS")
        print("="*80)
        
        for project_name, results in self.correlation_results.items():
            print(f"\nPROJECT: {project_name}")
            print("-" * 50)
            
            # Riassunto significatività
            all_results = []
            for category in ['complexity', 'changes', 'bugfixes']:
                if category in results:
                    for test_name, test_result in results[category].items():
                        all_results.append({
                            'category': category,
                            'test': test_name,
                            'r': test_result['correlation'],
                            'p': test_result['p_value'],
                            'significant': test_result['significant'],
                            'n': test_result['n']
                        })
            
            if all_results:
                df_results = pd.DataFrame(all_results)
                
                print(f"Total tests performed: {len(df_results)}")
                print(f"Significant correlations (p<0.05): {sum(df_results['significant'])}")
                print(f"Significant percentage: {sum(df_results['significant'])/len(df_results)*100:.1f}%")
                
                # Strongest correlations per category
                for category in ['complexity', 'changes', 'bugfixes']:
                    cat_data = df_results[df_results['category'] == category]
                    if not cat_data.empty:
                        strongest = cat_data.loc[cat_data['r'].abs().idxmax()]
                        print(f"\nStrongest {category} correlation:")
                        print(f"  {strongest['test']}: r={strongest['r']:.3f}, p={strongest['p']:.3f}")
    
    def export_results(self, filename_prefix="ml_cs_correlation"):
        """Esporta i risultati in formato JSON e CSV"""
        # Export correlations as JSON
        with open(f"{filename_prefix}_results.json", 'w') as f:
            json.dump(self.correlation_results, f, indent=2, default=str)
        
        # Export aggregated data as CSV for each project
        for project_name, data in self.aggregated_data.items():
            data.to_csv(f"{filename_prefix}_{project_name}_aggregated.csv", index=False)
        
        print(f"Results exported to {filename_prefix}_results.json and CSV files")

# Esempio di utilizzo
def main():
    # Inizializza l'analyzer
    analyzer = MLCodeSmellCorrelationAnalyzer({})
    
    # Lista dei tuoi progetti (sostituisci con i nomi reali)
    projects = [
        "project1", "project2", "project3", "project4", "project5", "project6"
    ]
    
    # Carica i dati per ogni progetto
    for project in projects:
        try:
            analyzer.load_project_data(
                project,
                f"{project}_commit_metrics.json",
                f"{project}_file_frequencies.json",
                f"{project}_smell_evolution.json"
            )
            print(f"Loaded data for {project}")
        except FileNotFoundError as e:
            print(f"File not found for {project}: {e}")
            continue
    
    # Esegui l'analisi per ogni progetto
    for project in analyzer.projects_data.keys():
        print(f"\nProcessing {project}...")
        
        # Aggrega i dati per mese
        analyzer.aggregate_data_by_time(project, window='M')
        
        # Esegui l'analisi di correlazione
        analyzer.perform_correlation_analysis(project)
    
    # Genera report riassuntivo
    analyzer.generate_summary_report()
    
    # Esporta risultati
    analyzer.export_results()

if __name__ == "__main__":
    main()