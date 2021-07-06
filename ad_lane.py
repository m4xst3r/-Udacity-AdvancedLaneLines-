import numpy as np

class Lane():
    def __init__(self):
        #determines if the line was detected in the frame before
        self.detected = False
        self.confident_cnt = 0

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
        self.curv_radius_left = None
        self.curv_radius_right = None

        #vehicle position regarding centre
        self.vehicle_pos = None
        self.vehicle_dir = None

        #last fitted line point - only x values as y values never fit
        self.fitted_points_list = []
        self.fitted_points_list_left = []
        self.fitted_points_list_right = []
        #average of the fitted points
        self.best_fit_x_left = []
        self.best_fit_x_right = None

    def smoothing_poly(self, frames= 4):
        """
        calculates an average from all poly values with given frame number
        returns True if smoothin is valid

        Args:
            frames (int, optional): [description]. Defaults to 4.
        """
        if len(self.fitted_points_list) < frames:
            self.fitted_points_list.append((self.left_fit, self.right_fit))
            return False
        else:
            self.fitted_points_list.append((self.left_fit, self.right_fit))
            temp_best_fit_x_left = []
            temp_best_fit_x_right = []
            for fitted_points in self.fitted_points_list:
                temp_best_fit_x_left.append(fitted_points[0])
                temp_best_fit_x_right.append(fitted_points[1])
            self.best_fit_x_left = np.sum(temp_best_fit_x_left, axis=0)/(frames+1)
            self.best_fit_x_right = np.sum(temp_best_fit_x_right, axis=0)/(frames+1)
            self.fitted_points_list.pop(0)
            return True

    def sanity_horizon(self):
        """
        check if the distnace between the lines is similar
        """

        diff = self.left_fit_line[0] - self.right_fit_line[0]

        mean = np.mean(np.absolute(diff))
    
        if np.absolute(diff) <= 450:
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
            return False
        else:
            return True

    def sanity_curv(self):
        """
        check if the curvature fo the lines is similar
        """

        diff = self.curv_radius_left - self.curv_radius_right

        if np.absolute(diff) > 1000:
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
                #only validate curvature of both lanes have a sane curvature
                if (self.curv_radius_left < 5000) and (self.curv_radius_right < 5000):
                    ret = self.sanity_curv()
                    if ret == True:
                        return True
                    else:
                        return False
                else:
                    return True
                   