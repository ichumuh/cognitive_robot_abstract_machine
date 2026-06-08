from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas
import pandas as pd

from giskardpy.qp.qp_data_symbolic import QPDataSymbolic
from giskardpy.utils.utils import create_path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

date_str = datetime.datetime.now().strftime("%Yy-%mm-%dd--%Hh-%Mm-%Ss")


@dataclass
class QuadraticProgramDebugger:
    """
    This class is designed to help you debug Giskard's quadratic programs (QP) by using names of constraints and
    degrees of freedom to create panda arrays with names rows and columns.
    """

    qp_data_symbolic: QPDataSymbolic
    """
    The symbolic casadi expressions for computing the QP components.
    """
    current_solution: np.ndarray | None = field(default=None)
    """
    The solution of the QP, None if there is none.
    """
    direct_limits: pandas.DataFrame = field(init=False)
    """
    This panda array gives you insights into the decision variables of the QP.
    It contains columns for direct upper and lower bounds, last solution and weights.
    """
    equality_constraints: pandas.DataFrame = field(init=False)
    """
    This panda array gives insight in the equality constraints.
    It contains columns for the equality bounds, the result of the equality matrix * decision variables (without slack),
    and the slack, which is essentially how much the constraints are violated. 
    """
    equality_matrix: pandas.DataFrame = field(init=False)
    """
    
    """
    inequality_constraints: pandas.DataFrame = field(init=False)
    inequality_matrix: pandas.DataFrame = field(init=False)

    def __post_init__(self):
        self.update(self.current_solution)

    def update(self, current_solution: np.ndarray):
        self.current_solution = current_solution
        last_solution = (
            np.ones(self.qp_data_symbolic.box_lower_constraints.shape[0]) * np.nan
        )
        if self.current_solution is not None:
            last_solution[self.quadratic_weight_filter] = self.current_solution

        self.current_solution = last_solution
        self.create_direct_limits()
        self.create_equality_constraints()
        self.create_inequality_constraints()

    @property
    def quadratic_weight_filter(self) -> np.ndarray:
        quadratic_weight_filter = np.ones(
            self.qp_data_symbolic.quadratic_weights.shape[0]
        )
        quadratic_weight_filter[self.qp_data_symbolic.num_non_slack_variables :] = (
            self.qp_data_symbolic.quadratic_weights.evaluate()[
                self.qp_data_symbolic.num_non_slack_variables :
            ]
            != 0
        )
        return quadratic_weight_filter.astype(bool)

    def create_direct_limits(self):
        self.direct_limits = pd.DataFrame(
            {
                "lower bounds": self.qp_data_symbolic.box_lower_constraints.evaluate(),
                "solution": self.current_solution,
                "upper bounds": self.qp_data_symbolic.box_upper_constraints.evaluate(),
                "quadratic weight": self.qp_data_symbolic.quadratic_weights.evaluate(),
                "linear weight": self.qp_data_symbolic.linear_weights.evaluate(),
            },
            self.free_variable_names,
            dtype=float,
        )

    def create_equality_constraints(self):
        eq_matrix_dofs_np = self.qp_data_symbolic.eq_matrix_dofs.evaluate()
        eq_matrix_slack_np = self.qp_data_symbolic.eq_matrix_slack.evaluate()
        Ex = eq_matrix_dofs_np @ self.current_solution[: eq_matrix_dofs_np.shape[1]]
        bounds = self.qp_data_symbolic.eq_bounds.evaluate()
        self.equality_constraints = pd.DataFrame(
            {
                "Ex": Ex,
                "slack": bounds - Ex,
                "bounds": bounds,
            },
            self.equality_constr_names,
            dtype=float,
        )
        self.equality_matrix = pd.DataFrame(
            eq_matrix_dofs_np,
            self.equality_constr_names,
            self.degree_of_freedom_names,
            dtype=float,
        )

    def create_inequality_constraints(self):
        neq_matrix_dofs_np = self.qp_data_symbolic.neq_matrix_dofs.evaluate()
        neq_matrix_slack_np = self.qp_data_symbolic.neq_matrix_slack.evaluate()
        Ex = neq_matrix_dofs_np @ self.current_solution[: neq_matrix_dofs_np.shape[1]]
        lower_bounds = self.qp_data_symbolic.neq_lower_bounds.evaluate()
        upper_bounds = self.qp_data_symbolic.neq_upper_bounds.evaluate()
        if len(self.inequality_constr_names) > 0:
            self.inequality_constraints = pd.DataFrame(
                {
                    "lower_bounds": lower_bounds,
                    "Ax": Ex,
                    # "slack": bounds - Ex,
                    "upper_bounds": upper_bounds,
                },
                self.inequality_constr_names,
                dtype=float,
            )
            self.inequality_matrix = pd.DataFrame(
                neq_matrix_dofs_np,
                self.inequality_constr_names,
                self.degree_of_freedom_names,
                dtype=float,
            )

    def _has_nan(self):
        nan_entries = self.p_A[0].isnull().stack()
        row_col_names = nan_entries[nan_entries].index.tolist()
        pass

    def _print_pandas_array(self, array):
        import pandas as pd

        if len(array) > 0:
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                print(array)

    def save_pandas(
        self, dfs, names, path, time: float, folder_name: Optional[str] = None
    ):

        if folder_name is None:
            folder_name = ""
        folder_name = f"{path}/pandas/{folder_name}_{date_str}/{time}/"
        create_path(folder_name)
        for df, name in zip(dfs, names):
            csv_string = "name\n"
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                if df.shape[1] > 1:
                    for column_name, column in df.T.items():
                        zero_filtered_column = (
                            column.replace(0, np.nan)
                            .dropna(how="all")
                            .replace(np.nan, 0)
                        )
                        csv_string += zero_filtered_column.add_prefix(
                            column_name + "||"
                        ).to_csv(float_format="%.6f")
                else:
                    csv_string += df.to_csv(float_format="%.6f")
            file_name2 = f"{folder_name}{name}.csv"
            with open(file_name2, "w") as f:
                f.write(csv_string)

    @property
    def free_variable_names(self) -> list[str]:
        return self.qp_data_symbolic.free_variable_names

    @property
    def degree_of_freedom_names(self) -> list[str]:
        names = []
        for derivative in ["vel", "jerk"]:
            for k in range(self.qp_data_symbolic.config.prediction_horizon):
                if (
                    derivative == "vel"
                    and k > self.qp_data_symbolic.config.prediction_horizon - 3
                ):
                    continue
                for dof in self.qp_data_symbolic.degrees_of_freedom:
                    names.append(f"{dof.name}_{derivative}_k_{k}")
        return names

    @property
    def equality_constr_names(self):
        return self.qp_data_symbolic.eq_constraint_names

    @property
    def inequality_constr_names(self):
        return self.qp_data_symbolic.neq_constraint_names

    def _print_iis(self):
        import pandas as pd

        def print_iis_matrix(
            row_filter: np.ndarray,
            column_filter: np.ndarray,
            matrix: pd.DataFrame,
            bounds: pd.DataFrame,
        ):
            if len(row_filter) == 0:
                return
            filtered_matrix = matrix.loc[row_filter, column_filter]
            filtered_matrix["bounds"] = bounds.loc[row_filter]
            print(filtered_matrix)

        result = self.qp_controller.qp_solver.analyze_infeasibility()
        if result is None:
            logger.info(
                f"Can only compute possible causes with gurobi, "
                f"but current solver is {self.qp_controller.config.qp_solver_id.name}."
            )
            return
        lb_ids, ub_ids, eq_ids, lbA_ids, ubA_ids = result
        b_ids = lb_ids | ub_ids
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.width", None
        ):
            logger.info("Irreducible Infeasible Subsystem:")
            logger.info("  Free variable bounds")
            free_variables = self.p_lb[b_ids]
            free_variables["ub"] = self.p_ub[b_ids]
            free_variables = free_variables.rename(columns={"data": "lb"})
            print(free_variables)
            logger.info("  Equality constraints:")
            print_iis_matrix(eq_ids, b_ids, self.p_E, self.p_bE)
            logger.info("  Inequality constraint lower bounds:")
            print_iis_matrix(lbA_ids, b_ids, self.p_A, self.p_lbA)
            logger.info("  Inequality constraint upper bounds:")
            print_iis_matrix(ubA_ids, b_ids, self.p_A, self.p_ubA)
