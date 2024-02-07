import random
import copy
from dataclasses import dataclass


@dataclass
class Solver_Params:
    p0: list
    sigma: float
    tmax: int
    pc: float


@dataclass
class Solver_Results:
    xmin: list
    grademin: float
    all_xmin: list


class Evolutionary_algorithm:
    def __init__(self, qx):
        self._qx = qx

    def stop_conditions(self, tmax, t) -> bool:
        if t >= tmax:
            return True
        return False

    def grades(self, f, population):
        grades = []
        for x in population:
            grades.append(abs(f(x)))
        return grades

    def find_best(self, population, grades):
        xmin = population[0]
        grademin = grades[0]
        for i, one in enumerate(grades[1:]):
            if one < grademin:
                xmin = population[i + 1]
                grademin = one
        return xmin, grademin

    def reproduction(self, population, grades, size):
        new_population = []
        for x in range(size):
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            turnament = [population[i], population[j]]
            grades_tur = [grades[i], grades[j]]
            best = self.find_best(turnament, grades_tur)
            new_population.append(best[0])
        return new_population

    def crossing(self, p1, pc):
        crossed_pop = []
        pop = copy.deepcopy(p1)
        for i in range(len(pop)):
            xval = random.randint(0, len(pop) - 1)
            yval = random.randint(0, len(pop) - 1)
            first = pop[xval]
            second = pop[yval]
            if random.gauss(0, 1) < pc:
                ival = random.randint(0, len(first) - 1)
                jval = random.randint(0, len(first) - 1)
                if jval > ival:
                    crossed = (
                        second[0:ival] + first[ival:jval] + second[jval : len(second)]
                    )
                else:
                    crossed = (
                        first[0:jval] + second[jval:ival] + first[ival : len(second)]
                    )
                crossed_pop.append(crossed)
            else:
                parents = [first, second]
                crossed_pop.append(parents[random.randint(0, 1)])
        return crossed_pop

    def mutation(self, p1, sigma):
        pop = copy.deepcopy(p1)
        new_population = []
        for x in pop:
            for i in range(len(x)):
                mutation = sigma * random.gauss(0, 1)
                x[i] = x[i] + mutation
            new_population.append(x)
        return new_population

    def elite_succession(self, pi, m_new, grades, m_grad):
        grp = zip(grades, pi)
        grp = sorted(grp)
        [new_grades, pop] = grp[0]
        mut_x = copy.deepcopy(m_new)
        mut_grad = copy.deepcopy(m_grad)
        mut_x.append(pop)
        mut_grad.append(new_grades)
        grp2 = zip(m_grad, m_new)
        grp2 = sorted(grp2)
        grades_new, pi_new = [
            [i for i, j in grp2[: len(pi)]],
            [j for i, j in grp2[: len(pi)]],
        ]
        return pi_new, grades_new

    def evolutionary_algorithm(self, params: Solver_Params) -> Solver_Results:
        all_xmin = []
        i = 0
        pi = params.p0
        grades1 = copy.deepcopy(self.grades(self._qx, pi))
        xmin, grademin = self.find_best(pi, grades1)
        all_xmin.append(xmin)
        while self.stop_conditions(params.tmax, i) is False:
            p_new = self.reproduction(pi, grades1, len(pi))
            c_new = self.crossing(p_new, params.pc)
            m_new = self.mutation(c_new, params.sigma)
            m_grad = self.grades(self._qx, m_new)
            xmin2, grademin2 = self.find_best(m_new, m_grad)
            if grademin2 < grademin:
                grademin = grademin2
                xmin = xmin2
                all_xmin.append(xmin)
            pi, grades1 = self.elite_succession(pi, m_new, grades1, m_grad)
            i += 1
        results = Solver_Results(xmin, grademin, all_xmin)
        return results
