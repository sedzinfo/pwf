import os
os.getcwd()

# Set the working directory
new_directory = "C:/Users/dzach/Desktop/examples"  # replace with your desired path
os.chdir(new_directory)

# Verify the change
print("New Working Directory:", os.getcwd())





from raven_gen import Matrix, MatrixType
import numpy as np
rpm = Matrix.make(matrix_type=np.random.choice(list(MatrixType)))
rpm.make_alternatives(n_alternatives=10)
rpm.save(".", "test")

print(rpm)
print(rpm.rules)

from raven_gen import Ruleset, RuleType
ruleset = Ruleset(size_rules=[RuleType.CONSTANT, RuleType.PROGRESSION], shape_rules=list(RuleType))
rpm = Matrix.make(np.random.choice(list(MatrixType)), ruleset=ruleset)
rpm.make_alternatives(n_alternatives=4)
rpm.save(".", "p2")

{ONE_SHAPE,FOUR_SHAPE,FIVE_SHAPE,NINE_SHAPE,TWO_SHAPE_VERTICAL_SEP,TWO_SHAPE_HORIZONTAL_SEP,SHAPE_IN_SHAPE,FOUR_SHAPE_IN_SHAPE}
{CONSTANT,PROGRESSION,ARITHMETIC,DISTRIBUTE_THREE}
{POSITION,NUMBER,SIZE,SHAPE,COLOR}

rpm = Matrix.make(matrix_type=raven_gen.matrix.MatrixType)
rpm.make_alternatives(n_alternatives=10)
rpm.save(".", "test")

make( matrix_type: raven_gen.matrix.MatrixType, ruleset: raven_gen.rule.Ruleset = None, n_alternatives: int = 0 )



from raven_gen import Ruleset, RuleType
ruleset = Ruleset(size_rules=[RuleType.CONSTANT, RuleType.PROGRESSION], shape_rules=list(RuleType))
rpm = Matrix.make(np.random.choice(list(MatrixType)), ruleset=ruleset)



import raven_gen
raven_gen.attribute.SIZE_VALUES
raven_gen.attribute.SIZE_VALUES = (10, 1, 1, 1, 1)
