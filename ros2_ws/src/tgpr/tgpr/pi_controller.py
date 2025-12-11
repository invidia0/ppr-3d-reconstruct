class PI:
    def __init__(self, kp=1.0, ki=0.1):
        """
        Define a PID controller class
        :param kp: float, kp coeff
        :param ki: float, ki coeff
        :param kd: float, kd coeff
        """
        self.kp = kp
        self.ki = ki
        self.Pterm = 0.0
        self.Iterm = 0.0
        self.last_error = 0.0

    def control(self, error, dt, cap=None):
        """
        PID main function, given an input, this function will output a control unit
        :param error: float, error term
        :param cap: float, maximum absolute value for output
        :return: float, output control
        """
        self.Pterm = self.kp * error
        self.Iterm += error * dt

        self.last_error = error
        output = self.Pterm + self.ki * self.Iterm
        # Cap the output
        if cap is not None:
            output = max(min(output, cap), -cap)
        return output