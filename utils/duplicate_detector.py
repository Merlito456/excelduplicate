import re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from collections import Counter

@dataclass
class Project:
    site_name: str
    plaid: str
    project: str
    additional_fields: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        # Clean and normalize the fields
        self.site_name = self._normalize_text(self.site_name)
        self.plaid = self._normalize_text(self.plaid)
        self.project = self._normalize_text(self.project)
        # Normalize additional fields
        if self.additional_fields:
            self.additional_fields = {
                k: self._normalize_text(v) for k, v in self.additional_fields.items()
            }
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text by removing extra spaces, converting to lowercase, etc."""
        if not text:
            return ""
        # Convert to lowercase and strip whitespace
        text = str(text).lower().strip()
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters for better matching
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def get_searchable_keywords(self) -> List[str]:
        """Generate searchable keywords from the project name."""
        words = self.project.split()
        # Remove common words and numbers for better matching
        filtered = [w for w in words if len(w) > 2 and not w.isdigit()]
        return filtered
    
    def get_all_text(self) -> str:
        """Get all text fields combined for overall similarity."""
        parts = [self.site_name, self.plaid, self.project]
        parts.extend(self.additional_fields.values())
        return ' '.join(parts)

class RowSimilarityAnalyzer:
    """Analyzes overall similarity between project rows."""
    
    def __init__(self):
        self.field_weights = {
            'site_name': 0.25,
            'plaid': 0.30,
            'project': 0.30,
            'additional': 0.15
        }
        self.min_field_match = 0.3  # Minimum score for a field to be considered matching
    
    def calculate_row_similarity(self, proj1: Project, proj2: Project) -> Dict[str, float]:
        """
        Calculate comprehensive similarity between two project rows.
        Returns detailed similarity scores for all fields and overall.
        """
        # Individual field similarities
        field_scores = {
            'site_name': self._calculate_similarity(proj1.site_name, proj2.site_name),
            'plaid': 1.0 if proj1.plaid == proj2.plaid else 0.0,
            'project': self._calculate_similarity(proj1.project, proj2.project)
        }
        
        # Additional fields similarity (if any)
        if proj1.additional_fields and proj2.additional_fields:
            common_keys = set(proj1.additional_fields.keys()) & set(proj2.additional_fields.keys())
            if common_keys:
                add_scores = []
                for key in common_keys:
                    score = self._calculate_similarity(
                        proj1.additional_fields.get(key, ''),
                        proj2.additional_fields.get(key, '')
                    )
                    add_scores.append(score)
                field_scores['additional'] = np.mean(add_scores) if add_scores else 0.0
            else:
                field_scores['additional'] = 0.0
        else:
            field_scores['additional'] = 0.0
        
        # Keyword overlap analysis
        keywords1 = set(proj1.get_searchable_keywords())
        keywords2 = set(proj2.get_searchable_keywords())
        
        if keywords1 and keywords2:
            overlap = len(keywords1.intersection(keywords2)) / max(len(keywords1), len(keywords2))
            field_scores['keyword_overlap'] = overlap
        else:
            field_scores['keyword_overlap'] = 0.0
        
        # N-gram similarity for overall text
        text1 = proj1.get_all_text()
        text2 = proj2.get_all_text()
        field_scores['text_similarity'] = self._calculate_similarity(text1, text2)
        
        # Weighted overall similarity
        weights = self.field_weights
        overall_score = sum(
            field_scores.get(field, 0) * weight 
            for field, weight in weights.items()
        )
        
        # Add keyword and text similarity as bonus
        overall_score += field_scores['keyword_overlap'] * 0.1
        overall_score += field_scores['text_similarity'] * 0.1
        
        # Normalize to 0-1 range
        overall_score = min(1.0, overall_score)
        field_scores['overall'] = overall_score
        
        # Calculate match significance
        field_scores['match_significance'] = self._calculate_match_significance(field_scores)
        
        return field_scores
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher."""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _calculate_match_significance(self, scores: Dict[str, float]) -> float:
        """
        Calculate how significant the match is based on multiple factors.
        Higher score means more confidence in the match.
        """
        # Count how many fields have good matches
        strong_matches = sum(1 for k, v in scores.items() 
                           if k != 'overall' and v >= self.min_field_match)
        
        # Weighted significance
        significance = 0.0
        if scores.get('plaid', 0) >= 0.8:
            significance += 0.4  # Plaid match is very strong
        if scores.get('site_name', 0) >= 0.7:
            significance += 0.25
        if scores.get('project', 0) >= 0.7:
            significance += 0.25
        if scores.get('keyword_overlap', 0) >= 0.5:
            significance += 0.1
        
        return min(1.0, significance)

class DuplicateDetector:
    def __init__(self, similarity_threshold: float = 0.75, 
                 row_similarity_threshold: float = 0.7,
                 weights: Optional[Dict[str, float]] = None):
        self.similarity_threshold = similarity_threshold
        self.row_similarity_threshold = row_similarity_threshold
        self.weights = weights or {
            'site_name': 0.2,
            'plaid': 0.3,
            'project': 0.3,
            'keyword_overlap': 0.2
        }
        self.projects: List[Project] = []
        self.row_analyzer = RowSimilarityAnalyzer()
        
    def load_from_dataframe(self, df: pd.DataFrame, additional_fields: List[str] = None) -> None:
        """Load projects from a pandas DataFrame with optional additional fields."""
        self.projects = []
        for _, row in df.iterrows():
            # Get additional fields if specified
            additional = {}
            if additional_fields:
                for field in additional_fields:
                    if field in row and field not in ['site_name', 'plaid', 'project']:
                        additional[field] = str(row[field])
            
            project = Project(
                site_name=str(row.get('site_name', '')),
                plaid=str(row.get('plaid', '')),
                project=str(row.get('project', '')),
                additional_fields=additional
            )
            self.projects.append(project)
    
    def load_from_csv(self, filepath: str, additional_fields: List[str] = None) -> None:
        """Load projects from a CSV file."""
        df = pd.read_csv(filepath)
        self.load_from_dataframe(df, additional_fields)
    
    def load_from_list(self, data: List[Dict[str, str]], additional_fields: List[str] = None) -> None:
        """Load projects from a list of dictionaries."""
        df = pd.DataFrame(data)
        self.load_from_dataframe(df, additional_fields)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher."""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _are_projects_similar(self, proj1: Project, proj2: Project) -> Tuple[bool, Dict[str, float]]:
        """
        Check if two projects are similar based on multiple criteria.
        Returns (is_similar, similarity_scores)
        """
        # Get comprehensive row similarity
        row_scores = self.row_analyzer.calculate_row_similarity(proj1, proj2)
        
        # Use row similarity as primary indicator
        is_similar = row_scores['overall'] >= self.row_similarity_threshold
        
        # Additional checks for special cases
        if not is_similar:
            # Check if projects share unique keywords
            keywords1 = set(proj1.get_searchable_keywords())
            keywords2 = set(proj2.get_searchable_keywords())
            
            # If they share significant keywords, they might be similar
            if keywords1 and keywords2:
                common = keywords1.intersection(keywords2)
                if len(common) >= 2:  # Share at least 2 meaningful words
                    is_similar = True
                elif len(common) >= 1 and row_scores['site_name'] >= 0.5:
                    is_similar = True
            
            # Check if one project name is a substring of the other
            if proj1.project and proj2.project:
                if proj1.project in proj2.project or proj2.project in proj1.project:
                    if row_scores['site_name'] >= 0.5 or row_scores['plaid'] >= 0.5:
                        is_similar = True
        
        # If marked as similar, verify with match significance
        if is_similar:
            # Ensure the match has enough evidence
            if row_scores['match_significance'] < 0.3:
                is_similar = False
        
        return is_similar, row_scores
    
    def find_duplicates(self) -> Dict[int, List[Tuple[int, float, Dict[str, float]]]]:
        """
        Find all duplicate projects based on the similarity criteria.
        Returns a dictionary mapping project index to list of (duplicate_index, similarity_score, scores_dict)
        """
        duplicates = {}
        seen_indices = set()
        processed_pairs = set()
        
        for i in range(len(self.projects)):
            if i in seen_indices:
                continue
                
            current_group = []
            for j in range(i + 1, len(self.projects)):
                pair_key = tuple(sorted([i, j]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                is_similar, scores = self._are_projects_similar(self.projects[i], self.projects[j])
                if is_similar:
                    current_group.append((j, scores['overall'], scores))
                    seen_indices.add(j)
            
            if current_group:
                duplicates[i] = current_group
                seen_indices.add(i)
        
        return duplicates
    
    def find_similar_rows_by_site(self, site_name: str) -> List[Tuple[int, Project, float]]:
        """
        Find all projects similar to a specific site.
        Returns list of (index, project, similarity_score)
        """
        results = []
        site_projects = [p for p in self.projects if site_name in p.site_name]
        
        if not site_projects:
            return results
        
        # Use the first matching project as reference
        reference = site_projects[0]
        
        for i, project in enumerate(self.projects):
            if project == reference:
                continue
            
            scores = self.row_analyzer.calculate_row_similarity(reference, project)
            if scores['overall'] >= self.row_similarity_threshold:
                results.append((i, project, scores['overall']))
        
        return sorted(results, key=lambda x: x[2], reverse=True)
    
    def get_similarity_matrix(self) -> pd.DataFrame:
        """
        Generate a similarity matrix for all projects.
        Returns a DataFrame with similarity scores.
        """
        n = len(self.projects)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                scores = self.row_analyzer.calculate_row_similarity(
                    self.projects[i], 
                    self.projects[j]
                )
                similarity = scores['overall']
                matrix[i][j] = similarity
                matrix[j][i] = similarity
        
        # Create labels
        labels = [f"{p.project[:20]}..." if len(p.project) > 20 else p.project 
                  for p in self.projects]
        
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        return df
    
    def get_cluster_analysis(self, duplicates: Dict = None) -> Dict[str, any]:
        """
        Analyze clusters of duplicates and provide statistics.
        """
        if duplicates is None:
            duplicates = self.find_duplicates()
        
        clusters = []
        all_duplicates = set()
        
        for main_idx, dup_list in duplicates.items():
            cluster = {
                'main_index': main_idx,
                'main_project': self.projects[main_idx],
                'duplicates': [self.projects[idx] for idx, _, _ in dup_list],
                'similarity_scores': [score for _, score, _ in dup_list],
                'size': len(dup_list) + 1
            }
            clusters.append(cluster)
            all_duplicates.add(main_idx)
            for idx, _, _ in dup_list:
                all_duplicates.add(idx)
        
        return {
            'total_clusters': len(clusters),
            'total_duplicates': len(all_duplicates),
            'unique_projects': len(self.projects) - len(all_duplicates),
            'clusters': clusters,
            'average_cluster_size': np.mean([c['size'] for c in clusters]) if clusters else 0,
            'max_cluster_size': max([c['size'] for c in clusters]) if clusters else 0
        }