import random
from dataclasses import dataclass


@dataclass
class Learning_params:
    lerning_rate: float
    tmax: int
    error: float


@dataclass
class Solver_Parameters:
    x0: list
    sigma0: float
    tmax: int
    interval: int


@dataclass
class Solver_Results:
    xmin: list
    allgrades: list
    successful: list
    suc_ind: list
    all_steps: list
    successful_x: list


class Evolutionary_algorithm:
    def __init__(self, qx, params: Solver_Parameters):
        self.params = params
        self.qx = qx

    def stop_conditions(self, t, suc_t) -> bool:
        tmax = self.params.tmax
        if t >= tmax or suc_t[-1] < (t - 30):
            return True
        return False

    def mutation(self, xin, sigma):
        x_mut = xin.copy()
        for i in range(len(x_mut)):
            mutation = sigma * random.gauss(0, 1)
            x_mut[i] = x_mut[i] + mutation
        return x_mut

    def run_algorithm(self) -> Solver_Results:
        grades = []
        successful = []
        suc_t = []
        all_x = []
        suc_x = []

        t = 1
        success_rate = 0
        xmin = self.params.x0
        grademin = self.qx(xmin)
        sigma = self.params.sigma0

        grades.append(grademin)
        all_x.append(xmin)
        suc_t.append(0)
        suc_x.append(xmin)

        while self.stop_conditions(t, suc_t) is False:
            m_new = self.mutation(xmin, sigma)
            m_grad = self.qx(m_new)

            grades.append(m_grad)
            all_x.append(m_new)

            if m_grad < grademin:
                grademin = m_grad
                xmin = m_new
                success_rate += 1

                successful.append(m_grad)
                suc_x.append(m_new)
                suc_t.append(t)
            if t % self.params.interval == 0:
                if (success_rate / self.params.interval) > 0.2:
                    sigma = 1.22 * sigma
                elif (success_rate / self.params.interval) < 0.2:
                    sigma = 0.82 * sigma
                success_rate = 0
            t += 1

        results = Solver_Results(xmin, grades, successful, suc_t, all_x, suc_x)
        return results
