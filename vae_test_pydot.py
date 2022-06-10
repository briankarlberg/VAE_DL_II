import sys
import pydot_ng as pydot

def main() -> int:
   pydot.Dot.create(pydot.Dot())
   print("Test Run Complete")

if __name__ == '__main__':
   sys.exit(main())