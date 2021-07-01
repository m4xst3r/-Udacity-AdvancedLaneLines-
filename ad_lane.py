import numpy as np

class Lane():
    def __init__(self):
        #determines if the line was detected in the frame before
        self.detected = False

        #lane points from the current frame
        self.leftx = None
        self.lefty = None
        self.rightx = None
        self.righty = None

        #values to use for the polynom function
        self.left_fit = None
        self.right_fit = None
        self.plot_points = None

        #fitted line using polynom
        self.left_fit_line = None
        self.right_fit_line = None

        #curvature calulated
        self.curv_radius = None

        #vehicle position regarding centre
        self.vehicle_pos = None
        self.vehicle_dir = None

    def sanity_horizon(self):
        """
        check if the distnace between the lines is similar
        """

        diff = self.left_fit_line[0] - self.right_fit_line[0]

        mean = np.mean(np.absolute(diff))
    
        if np.absolute(diff) <= 450:
            print('Horiz', diff)
            return False
        else:
            return True

    def sanity_parallel(self):
        """
        check if the detcetd lines are prallel // distance is always the same
        """
        diff = np.absolute(self.left_fit_line - self.right_fit_line)
        diff = np.diff(diff)
        
        #Check if the step is too high wich would mean the lines dirft apart
        if np.max(diff) >= 0.5:
            print('Para', diff)
            return False
        else:
            return True

    def sanity_check(self):
        """
        combine all checks in on fuction for easy handling
        """
        ret = self.sanity_horizon()
        if ret == False:
            return False
        else:
            ret = self.sanity_parallel()
            if ret == False:
                return False
            else:
                return True