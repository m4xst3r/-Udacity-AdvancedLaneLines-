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

