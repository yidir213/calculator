class Calculator :
    # return the sum between 2 operands
    def mysum(self, first_operand, second_operand):
        return first_operand + second_operand

    def min(self, first, second):
        if (first> second):
            return second
        return first

    def computeMax(self, first_operand, second_operand):
        return max(first_operand, second_operand)