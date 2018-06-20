from nonetworkx import Initialize, SecondStep
import pickle

if __name__ == "__main__":
    k = Initialize()

    with open("mat.pkl", "rb") as f:
        mat = pickle.load(f)

    SecondStep(mat, k)