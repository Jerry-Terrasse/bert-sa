import cProfile

from review import main

cProfile.run("main()", sort="tottime")