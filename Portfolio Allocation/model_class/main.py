from model_classes import *


if __name__ == "__main__":
    
    # Solve optimization multiple times

    for j in range(10):
        print()
        print("Starting model {}".format(j))
        print()

        m = MultStudentT("test_model_1")
        m.process_raw_data()
        m.train_test_split(perc=0.95)
        m.build_model()
        m.load_trace()
        m.compute_forecast()
        m.build_prob()
        m.solve_prob()
        print(m.opt_weights)

        del m


