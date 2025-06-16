#!/usr/bin/env python3
"""
ML Code Smells Correlation Analysis - Main Workflow
==================================================

Script principale per l'analisi di correlazione tra ML Code Smells e metriche di qualità.
Esegue automaticamente tutto il workflow dall'import dei dati ai risultati finali.

Usage:
    python main_workflow.py

File Structure Required:
    data/
    ├── project1/
    │   ├── commit_metrics.json
    │   ├── file_frequencies.json
    │   └── smell_evolution.json
    ├── project2/
    │   ├── commit_metrics.json
    │   ├── file_frequencies.json
    │   └── smell_evolution.json
    └── ...

Author: Simone Silvestri
Date: 2025-06-14
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.figure as Figure

# Import delle classi di analisi
from ml_cs_analyzer import MLCodeSmellCorrelationAnalyzer
from advanced_analyzer import AdvancedMLCSAnalysis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MLCSWorkflow:
    """
    Classe principale per orchestrare tutto il workflow di analisi
    """
    
    def __init__(self, data_dir="data", output_dir="results", time_window='M'):
        """
        Inizializza il workflow
        
        Args:
            data_dir (str): Directory contenente i dati dei progetti
            output_dir (str): Directory per salvare i risultati
            time_window (str): Finestra temporale per aggregazione ('M'=mese, 'Q'=trimestre, 'W'=settimana)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.time_window = time_window
        
        # Crea directory di output se non esiste
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inizializza analyzers
        self.base_analyzer = MLCodeSmellCorrelationAnalyzer({})
        self.advanced_analyzer = None
        
        # Traccia progetti processati
        self.processed_projects = []
        self.failed_projects = []
        self.project_figures = {}
        
        logger.info(f"Workflow initialized with data_dir={data_dir}, output_dir={output_dir}")
    
    def discover_projects(self):
        """
        Scopre automaticamente i progetti nella directory data
        
        Returns:
            list: Lista dei nomi dei progetti trovati
        """
        projects = []
        
        if not self.data_dir.exists():
            logger.error(f"Data directory {self.data_dir} does not exist!")
            return projects
        
        for item in self.data_dir.iterdir():
            if item.is_dir():
                # Verifica che abbia i file richiesti
                required_files = [
                    'commit_metrics.json',
                    'file_frequencies.json', 
                    'smell_evolution.json'
                ]
                
                has_all_files = all((item / file).exists() for file in required_files)
                
                if has_all_files:
                    projects.append(item.name)
                    logger.info(f"Found project: {item.name}")
                else:
                    missing_files = [f for f in required_files if not (item / f).exists()]
                    logger.warning(f"Project {item.name} missing files: {missing_files}")
        
        logger.info(f"Discovered {len(projects)} valid projects")
        return projects
    
    def load_project_data(self, project_name):
        """
        Carica i dati di un singolo progetto
        
        Args:
            project_name (str): Nome del progetto
            
        Returns:
            bool: True se caricamento riuscito, False altrimenti
        """
        try:
            project_dir = self.data_dir / project_name
            
            # Paths dei file
            commit_file = project_dir / 'commit_metrics.json'
            freq_file = project_dir / 'file_frequencies.json'
            evolution_file = project_dir / 'smell_evolution.json'
            
            # Carica i dati
            with open(commit_file, 'r', encoding='utf-8') as f:
                commit_data = json.load(f)
            
            with open(freq_file, 'r', encoding='utf-8') as f:
                freq_data = json.load(f)
            
            with open(evolution_file, 'r', encoding='utf-8') as f:
                evolution_data = json.load(f)
            
            # Valida i dati
            if not self._validate_data(project_name, commit_data, freq_data, evolution_data):
                return False
            
            # Salva nel base analyzer
            self.base_analyzer.projects_data[project_name] = {
                'commit_metrics': commit_data,
                'file_frequencies': freq_data,
                'smell_evolution': evolution_data
            }
            
            logger.info(f"Successfully loaded data for project: {project_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load project {project_name}: {str(e)}")
            return False
    
    def _validate_data(self, project_name, commit_data, freq_data, evolution_data):
        """
        Valida la struttura e qualità dei dati
        """
        try:
            # Valida commit_data
            if not isinstance(commit_data, list) or len(commit_data) == 0:
                logger.error(f"{project_name}: commit_metrics deve essere una lista non vuota")
                return False
            
            # Verifica campi richiesti nel primo commit
            required_fields = [
                'commit_hash', 'date', 'total_smells_found', 'smell_density',
                'project_cyclomatic_complexity', 'files_changed', 'is_bug_fix'
            ]
            
            first_commit = commit_data[0]
            missing_fields = [field for field in required_fields if field not in first_commit]
            
            if missing_fields:
                logger.error(f"{project_name}: Missing required fields in commit_metrics: {missing_fields}")
                return False
            
            # Valida evolution_data
            if 'summary' not in evolution_data:
                logger.error(f"{project_name}: smell_evolution must have 'summary' field")
                return False
            
            logger.info(f"{project_name}: Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"{project_name}: Data validation failed: {str(e)}")
            return False
    
    def run_base_analysis(self):
        """
        Esegue l'analisi di correlazione base per tutti i progetti
        """
        logger.info("Starting base correlation analysis...")
        
        successful_projects = []
        
        for project_name in self.base_analyzer.projects_data.keys():
            try:
                logger.info(f"Processing project: {project_name}")
                
                # Aggrega i dati per finestra temporale
                aggregated_data = self.base_analyzer.aggregate_data_by_time(project_name, self.time_window)
                
                if len(aggregated_data) < 3:
                    logger.warning(f"{project_name}: Insufficient data points ({len(aggregated_data)}) for correlation analysis")
                    self.failed_projects.append(project_name)
                    continue
                
                # Esegui analisi di correlazione
                correlation_results = self.base_analyzer.perform_correlation_analysis(project_name)

                successful_projects.append(project_name)
                self.processed_projects.append(project_name)
                
                logger.info(f"Completed base analysis for: {project_name}")
                
            except Exception as e:
                logger.error(f"Failed base analysis for {project_name}: {str(e)}")
                self.failed_projects.append(project_name)
        
        logger.info(f"Base analysis completed. Success: {len(successful_projects)}, Failed: {len(self.failed_projects)}")
        return successful_projects
    
    def run_advanced_analysis(self):
        """
        Esegue l'analisi avanzata se ci sono risultati interessanti
        """
        if not self.processed_projects:
            logger.warning("No projects available for advanced analysis")
            return
        
        logger.info("Starting advanced analysis...")
        
        # Inizializza advanced analyzer
        self.advanced_analyzer = AdvancedMLCSAnalysis(self.base_analyzer)
        
        try:
            # 1. Lag correlation analysis
            logger.info("Running lag correlation analysis...")
            for project in self.processed_projects:
                self.advanced_analyzer.lag_correlation_analysis(project, max_lag=3)
            
            # 2. Cross-project meta-analysis
            logger.info("Running cross-project meta-analysis...")
            meta_results = self.advanced_analyzer.cross_project_meta_analysis()
            
            # 3. Temporal stability analysis
            logger.info("Running temporal stability analysis...")
            self.advanced_analyzer.temporal_stability_analysis()
            
            # 4. Smell type-specific analysis
            logger.info("Running smell type-specific analysis...")
            self.advanced_analyzer.smell_type_specific_analysis()
            
            logger.info("Advanced analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Advanced analysis failed: {str(e)}")
    
    def generate_final_report(self):
        """
        Genera il report finale con tutti i risultati
        """
        logger.info("Generating final report...")
        
        try:
            # Generate summary report
            print("\n" + "="*80)
            print("FINAL ANALYSIS REPORT")
            print("="*80)
            print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Time Window: {self.time_window}")
            print(f"Total Projects Analyzed: {len(self.processed_projects)}")
            print(f"Failed Projects: {len(self.failed_projects)}")
            
            if self.failed_projects:
                print(f"Failed Projects List: {', '.join(self.failed_projects)}")
            
            # Base analysis summary
            self.base_analyzer.generate_summary_report()
            
            # Research questions answers
            self._answer_research_questions()
            
            logger.info("Final report generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate final report: {str(e)}")
    
    def _answer_research_questions(self):
        """
        Risponde direttamente alle 3 domande di ricerca
        """
        print("\n" + "="*80)
        print("RESEARCH QUESTIONS ANSWERS")
        print("="*80)
        
        # Analizza tutti i risultati per rispondere alle domande
        if not hasattr(self.base_analyzer, 'correlation_results'):
            print("No correlation results available for analysis")
            return
        
        # Question 1: ML-CSs vs Complexity
        print("\n1. Is the presence of ML-CSs correlated with code complexity increase over time?")
        print("-" * 70)
        
        complexity_results = []
        for project, results in self.base_analyzer.correlation_results.items():
            if 'complexity' in results:
                for test_name, test_result in results['complexity'].items():
                    if test_result['significant']:
                        complexity_results.append({
                            'project': project,
                            'correlation': test_result['correlation'],
                            'p_value': test_result['p_value']
                        })
        
        if complexity_results:
            avg_correlation = sum(r['correlation'] for r in complexity_results) / len(complexity_results)
            significant_count = len(complexity_results)
            total_projects = len(self.processed_projects)
            
            print(f"ANSWER: YES - Significant correlation found in {significant_count}/{total_projects} projects")
            print(f"Average correlation coefficient: {avg_correlation:.3f}")
            print("INTERPRETATION: ML Code Smells are positively correlated with complexity increases")
        else:
            print("ANSWER: NO - No significant correlation found between ML-CSs and complexity")
        
        # Question 2: ML-CSs vs Changes
        print("\n2. Is the presence of ML-CSs correlated with change activity increase over time?")
        print("-" * 70)
        
        change_results = []
        for project, results in self.base_analyzer.correlation_results.items():
            if 'changes' in results:
                for test_name, test_result in results['changes'].items():
                    if test_result['significant']:
                        change_results.append({
                            'project': project,
                            'correlation': test_result['correlation'],
                            'p_value': test_result['p_value']
                        })
        
        if change_results:
            avg_correlation = sum(r['correlation'] for r in change_results) / len(change_results)
            significant_count = len(change_results)
            total_projects = len(self.processed_projects)
            
            print(f"ANSWER: YES - Significant correlation found in {significant_count}/{total_projects} projects")
            print(f"Average correlation coefficient: {avg_correlation:.3f}")
            print("INTERPRETATION: ML Code Smells are associated with increased change activity")
        else:
            print("ANSWER: NO - No significant correlation found between ML-CSs and change activity")
        
        # Question 3: ML-CSs vs Bug Fixes
        print("\n3. Is the presence of ML-CSs correlated with bug fix commit activities over time?")
        print("-" * 70)
        
        bugfix_results = []
        for project, results in self.base_analyzer.correlation_results.items():
            if 'bugfixes' in results:
                for test_name, test_result in results['bugfixes'].items():
                    if test_result['significant']:
                        bugfix_results.append({
                            'project': project,
                            'correlation': test_result['correlation'],
                            'p_value': test_result['p_value']
                        })
        
        if bugfix_results:
            avg_correlation = sum(r['correlation'] for r in bugfix_results) / len(bugfix_results)
            significant_count = len(bugfix_results)
            total_projects = len(self.processed_projects)
            
            print(f"ANSWER: YES - Significant correlation found in {significant_count}/{total_projects} projects")
            print(f"Average correlation coefficient: {avg_correlation:.3f}")
            print("INTERPRETATION: ML Code Smells are associated with increased bug fix activities")
        else:
            print("ANSWER: NO - No significant correlation found between ML-CSs and bug fix activities")
    
    def export_all_results(self):
        """
        Esporta tutti i risultati in files strutturati
        """
        logger.info("Exporting all results...")
        
        try:
            # Create timestamped output directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = self.output_dir / f"analysis_{timestamp}"
            export_dir.mkdir(exist_ok=True)
            

            # Export visualizations
            self.base_analyzer.create_visualizations(str(export_dir))
            self.advanced_analyzer.create_advanced_visualizations(str(export_dir))


            # Export base analysis results
            base_prefix = str(export_dir / "base_analysis")
            self.base_analyzer.export_results(base_prefix)
            
            # Export advanced analysis results if available
            if self.advanced_analyzer:
                advanced_prefix = str(export_dir / "advanced_analysis")
                self.advanced_analyzer.export_advanced_results(advanced_prefix)
            
            # Export workflow summary
            workflow_summary = {
                'analysis_date': datetime.now().isoformat(),
                'time_window': self.time_window,
                'processed_projects': self.processed_projects,
                'failed_projects': self.failed_projects,
                'total_projects': len(self.processed_projects) + len(self.failed_projects)
            }
            
            with open(export_dir / "workflow_summary.json", 'w') as f:
                json.dump(workflow_summary, f, indent=2)
            
            # Copy log file
            if Path('analysis.log').exists():
                import shutil
                shutil.copy('analysis.log', export_dir / 'analysis.log')
            
            logger.info(f"All results exported to: {export_dir}")
            print(f"\n✅ All results exported to: {export_dir}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")
    
    def run_complete_workflow(self):
        """
        Esegue l'intero workflow dall'inizio alla fine
        """
        logger.info("="*60)
        logger.info("STARTING COMPLETE ML CODE SMELLS ANALYSIS WORKFLOW")
        logger.info("="*60)
        
        try:
            # Step 1: Discover projects
            projects = self.discover_projects()
            if not projects:
                logger.error("No valid projects found. Exiting.")
                return False
            
            # Step 2: Load all project data
            logger.info("Loading project data...")
            for project in projects:
                success = self.load_project_data(project)
                if not success:
                    self.failed_projects.append(project)
            
            if not self.base_analyzer.projects_data:
                logger.error("No projects loaded successfully. Exiting.")
                return False
            
            # Step 3: Run base analysis
            successful_projects = self.run_base_analysis()
            
            if not successful_projects:
                logger.error("No projects completed base analysis. Exiting.")
                return False
            
            # Step 4: Run advanced analysis (if there are enough successful projects)
            if len(successful_projects) >= 2:
                self.run_advanced_analysis()
            else:
                logger.warning("Skipping advanced analysis (need at least 2 successful projects)")
            
            # Step 5: Generate final report
            self.generate_final_report()
            
            # Step 6: Export all results
            self.export_all_results()
            
            logger.info("="*60)
            logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            return False

def main():
    """
    Funzione principale per eseguire il workflow
    """
    print("ML Code Smells Correlation Analysis - Workflow")
    print("=" * 50)
    
    # Configurazione (puoi modificare questi parametri)
    config = {
        'data_dir': 'data',           # Directory con i dati dei progetti
        'output_dir': 'results',      # Directory per i risultati
        'time_window': 'Q'            # Finestra temporale: 'M'=mese, 'Q'=trimestre, 'W'=settimana
    }
    
    # Inizializza e esegui workflow
    workflow = MLCSWorkflow(**config)
    success = workflow.run_complete_workflow()
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        print("Check the 'results' directory for all outputs.")
    else:
        print("\n❌ Analysis failed. Check 'analysis.log' for details.")
    
    return success

if __name__ == "__main__":
    main()