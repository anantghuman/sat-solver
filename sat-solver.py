from collections import OrderedDict


class sat_solver:
    def __init__(self):
        print("SAT Solver")
        cnf_file = input("Enter the path to the CNF file: ")
        try:
            with open(cnf_file, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:
            print("Wrong file or file path")
        self.clauses = []

        self.num_vars = 0
        self.num_clauses = 0
        for line in lines:
            # skip comments and empty lines
            line = line.strip()
            if line.startswith('c') or not line:
                continue
            if line.startswith('p cnf'):
                l = line.split()
                self.num_vars = int(l[2])
                self.num_clauses = int(l[3])
                continue
            l = line.split()
            clause = []
            for i in range(len(l) - 1):
                clause.append(int(l[i]))
            self.clauses.append(clause)
        """
        if len(clauses) != num_clauses:
            print("Error: Number of clauses does not match the header")
            return
        if num_vars == 0 or num_clauses == 0:
            # print("Error: Number of variables or clauses is zero")
            return
        print("Number of variables:", num_vars)
        print("Number of clauses:", num_clauses)
        print("Parsed CNF clauses:")
        for clause in self.clauses:
            print(clause)
        """
        self.assignments = {}
        self.frequency = {}
        # initialize the assignments and frequency
        for clause in self.clauses:
            for literal in clause:
                if abs(literal) not in self.assignments:
                    self.assignments[abs(literal)] = None
                if abs(literal) not in self.frequency:
                    self.frequency[abs(literal)] = 1
                else:
                    self.frequency[abs(literal)] += 1
            # if clause is a unit clause set the assignment
            if len(clause) == 1:
                self.assignments[abs(clause[0])] = True if clause[0] > 0 else False
        self.frequency = OrderedDict(sorted(self.frequency.items(), reverse = True))
    



    def dpll(self):
        for literal in self.frequency.keys():
            if self.assignments[literal] is not None:
                continue
            if not self.unit_propagation(self):
                return False
            if self.assignments[literal] is None:
                self.assignments[literal] = True
                break
    

    def unit_propagation(self):
        found_unit_clause = True
        while found_unit_clause:
            found_unit_clause = False
            for clause in self.clauses:
                unassigned_literals = []
                for literal in clause:
                    if abs(literal) not in self.assignments:
                        unassigned_literals.append(literal)
                    elif self.assignments[abs(literal)] is False and literal > 0:
                        break
                    elif self.assignments[abs(literal)] is True and literal < 0:
                        break
                if unassigned_literals is None:
                    continue
                if len(unassigned_literals) == 1:
                    found_unit_clause = True
                    unit_literal = unassigned_literals[0]
                    if self.assignments[abs(unit_literal)] is None:
                        self.assignments[abs(unit_literal)] = True if unit_literal > 0 else False
                    else:
                        return False
    
        return True







    def solve(self):
        dpll_result = self.dpll()

def main():
    solver = sat_solver()
    solver.solve()

if __name__ == "__main__":
    main()