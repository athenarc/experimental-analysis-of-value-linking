from typing import Dict, List

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.sql_meta_info import SQLMetaInfo
from workflow.agents.tool import Tool

class Evaluate(Tool):
    """
    Tool for evaluating the generated unit tests.
    """

    def __init__(self, template_name: str = None, engine_config: str = None, parser_name: str = None):
        super().__init__()
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name
        self.scores = []
        self.comparison_matrix = []
        self.SQL_id = None
        

    def _run(self, state: SystemState):
        """
        Executes the unit test evaluation process.
        
        Args:
            state (SystemState): The current system state.
        """
        # --- CORRECTED LOGIC ---

        # 1. Initialize variables to safe default values BEFORE the try block.
        target_SQL_meta_infos = []
        key_to_evaluate = ""

        # 2. Try to populate the variables from the system state.
        try:
            key_to_evaluate = list(state.SQL_meta_infos.keys())[-1]
            target_SQL_meta_infos = state.SQL_meta_infos[key_to_evaluate]
        except Exception as e:
            print(f"Error in UnitTestEvaluator while getting target SQLs: {e}. Proceeding with empty list.")

        # 3. Determine the ID for the new SQL entry.
        if key_to_evaluate and key_to_evaluate.startswith(self.tool_name):
            id = int(key_to_evaluate[len(self.tool_name)+1:])
            self.SQL_id = self.tool_name + "_" + str(id+1)
        else:
            self.SQL_id = self.tool_name + "_1"
        
        state.SQL_meta_infos[self.SQL_id] = []
        
        # 4. Now, safely handle the different cases based on the contents of target_SQL_meta_infos.
        # This block now correctly handles the case where the try block failed.
        if len(target_SQL_meta_infos) == 0:
            placeholder_sql = SQLMetaInfo(SQL="SELECT * FROM table_name")
            state.SQL_meta_infos[self.SQL_id].append(placeholder_sql)
            self.scores = [0]
            self.comparison_matrix = [[0]]
            return

        if len(target_SQL_meta_infos) == 1:
            state.SQL_meta_infos[self.SQL_id].append(target_SQL_meta_infos[0])
            self.scores = [1]
            self.comparison_matrix = [[1]]
            return

        if len(state.unit_tests["unit_test_generation"]) == 0:
            state.SQL_meta_infos[self.SQL_id].append(target_SQL_meta_infos[0])
            self.scores = [1]
            self.comparison_matrix = [[1]]
            return

        # --- THE REST OF THE FUNCTION CONTINUES AS BEFORE ---
        request_list = []
        candidates_clusters = self.execution_based_clustering(target_SQL_meta_infos)
        formatted_candidates = ""
        for index, candidate_query in enumerate(target_SQL_meta_infos):
            formatted_candidates += f"Candidate Response #{index+1}: Query: {candidate_query.SQL}\n, Execution Result: {self._format_sql_query_result(candidate_query)}\n"
        database_schema = state.get_database_schema_for_queries(
                [sql_meta_info.SQL for sql_meta_info in target_SQL_meta_infos]
            )
        for index, unit_test in enumerate(state.unit_tests["unit_test_generation"]): 
            try:            
                request_kwargs = {
                    "DATABASE_SCHEMA": database_schema,
                    "QUESTION": state.task.question,
                    "HINT": state.task.evidence,
                    "CANDIDATE_RESPONSES": formatted_candidates,
                    "UNIT_TEST": unit_test
                }
                request_list.append(request_kwargs)
            except Exception as e:
                print(f"Error in UnitTestEvaluator while creating request list: {e}")
                continue
                
        try:
            response = async_llm_chain_call(
                prompt=get_prompt(template_name=self.template_name),
                engine=get_llm_chain(**self.engine_config),
                parser=get_parser(self.parser_name),
                request_list=request_list,
                step=self.tool_name,
                # --- START OF FIX ---
                engine_config=self.engine_config
                # --- END OF FIX ---
            )
            response = [r[0] for r in response if r]
        except Exception as e:
            print(f"Error in Checker while getting response: {e}")
            response = []

        # Add a check for an empty comparison_matrix to prevent crash
        if not response:
            # If LLM fails or returns nothing, fall back to picking the first candidate
            state.SQL_meta_infos[self.SQL_id].append(target_SQL_meta_infos[0])
            self.scores = [1] + [0] * (len(target_SQL_meta_infos) - 1)
            self.comparison_matrix = [[1] + [0] * (len(target_SQL_meta_infos) - 1)]
            return

        comparison_matrix = []
        for item in response:
            if item and "scores" in item:
                comparison_matrix.append(item["scores"])

        # Add another check for an empty comparison_matrix
        if not comparison_matrix or not comparison_matrix[0]:
            state.SQL_meta_infos[self.SQL_id].append(target_SQL_meta_infos[0])
            self.scores = [1] + [0] * (len(target_SQL_meta_infos) - 1)
            self.comparison_matrix = [[1] + [0] * (len(target_SQL_meta_infos) - 1)]
            return

        self.comparison_matrix = comparison_matrix  
        scores = [sum([score[index] for score in comparison_matrix]) for index in range(len(comparison_matrix[0]))]
        self.scores = scores
        best_candidate = self.pick_the_best_candidate(scores, target_SQL_meta_infos, candidates_clusters)
        state.SQL_meta_infos[self.SQL_id].append(best_candidate)


    def self_consistency(self, candidate_clusters: Dict) -> SQLMetaInfo:
        """
        picks the candidate with the largest cluster.
        
        Args:
            candidates_clusters (Dict): The clusters of the candidates.
        """
        largest_cluster = max(candidate_clusters, key=lambda x: len(candidate_clusters[x]))
        return candidate_clusters[largest_cluster][0]
    

    def test_case_filtering_based_on_inter_cluster_variance(
            self,
            candidates_clusters: Dict,
            scores: List[int],
            target_SQL_meta_infos: List[SQLMetaInfo]
    ) -> bool:
        """
        Filters the test cases based on the inter-cluster variance.
        
        Args:
            candidates_clusters (Dict): The clusters of the candidates.
            scores (List[int]): The scores of the candidates.
            target_SQL_meta_infos (List[SQLMetaInfo]): The target SQL meta information.
        """
        for key, candidates in candidates_clusters.items():
            cluster_scores = [scores[target_SQL_meta_infos.index(candidate)] for candidate in candidates]
            if len(set(cluster_scores)) > 1:
                return False
        return True

    def pick_the_best_candidate(self, scores: List[int], candidates: List[SQLMetaInfo], candidate_clusters: Dict) -> SQLMetaInfo:
        """
        Picks the best candidate based on the scores.
        
        Args:
            scores (List[int]): The scores of the candidates.
            candidates (List[SQLMetaInfo]): The candidates.
            candidate_clusters (Dict): The clusters of the candidates.
        """
        largest_cluster = max(candidate_clusters, key=lambda x: len(candidate_clusters[x]))
        max_score = max(scores)
        best_candidates = [candidates[index] for index, score in enumerate(scores) if score == max_score]
        if len(best_candidates) == 1:
            return best_candidates[0]
        for candidate in best_candidates:
            if candidate in candidate_clusters[largest_cluster]:
                return candidate
        return best_candidates[0]
    

    def _format_sql_query_result(self, sql_meta_info: SQLMetaInfo) -> str:
        """
        Formats the SQL query to pass to the picker model.
        
        Args:
            sql_meta_info (SQLMetaInfo): The SQL meta information.
        """
        try:
            execution_result = sql_meta_info.execution_result
            if execution_result is None:
                return "No results"
            if not isinstance(execution_result, list):
                execution_result = list(execution_result)
            number_of_rows = len(execution_result)
            if number_of_rows == 0:
                number_of_columns = 0
            else:
                number_of_columns = len(execution_result[0])
            if number_of_rows > 20:
                execution_result = execution_result[:20]
            formatted_result = (
                f"Rows: {number_of_rows}, Columns: {number_of_columns}, Results:"
                f" {execution_result}"
            )
        except Exception as e:
            formatted_result = f"Error: {e}"
        return formatted_result
    
    def execution_based_clustering(self, candidate_queries: List[SQLMetaInfo]) -> list:
        """
        Clusters the generated candidates based on the execution results.
        
        Args:
            state (SystemState): The current system state.
        """
        clusters = {}
        for query in candidate_queries:
            try:
                result = str(query.execution_result) if isinstance(query.execution_result, str) else repr(query.execution_result)
            except Exception:
                continue
            if result not in clusters:
                clusters[result] = []
            clusters[result].append(query)
        # sample one query from each cluster
        return clusters
        
    def _get_updates(self, state: SystemState) -> Dict:
        key_to_evaluate = list(state.SQL_meta_infos.keys())[-2]
        target_SQL_meta_infos = state.SQL_meta_infos[key_to_evaluate]
        return {
            "scores": self.scores,
            "comparison_matrix": self.comparison_matrix,
            "candidates": [sql_meta_info.SQL for sql_meta_info in target_SQL_meta_infos],
            "selected_candidate": state.SQL_meta_infos[self.SQL_id][0].SQL
        }