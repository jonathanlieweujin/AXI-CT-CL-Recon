import numpy
from scipy import special, ndimage


class MaskGenerator:
    r'''
    Detects outliers in a numpy array and returns a boolean mask, True where outliers were detected,
    False for other pixels. Use one of the static constructors to configure a generator for your needs.
    '''

    @staticmethod
    def special_values(nan=True, inf=True):
        r'''This creates a MaskGenerator which generates a mask for inf and/or nan values.

        :param nan: mask NaN values
        :type nan: bool, default=True
        :param inf: mask INF values
        :type inf: bool, default=True

        '''
        if nan is True:
            if inf is True:
                generator = MaskGenerator(mode='special_values')
            else:
                generator = MaskGenerator(mode='nan')
        else:
            if inf is True:
                generator = MaskGenerator(mode='inf')
            else:
                raise ValueError("Please specify at least one type of value to threshold on")

        return generator

    @staticmethod
    def threshold(min_val=None, max_val=None):
        r'''This creates a MaskGenerator which generates a mask for values outside boundaries

        :param min_val: lower boundary
        :type min_val: float, default=None
        :param max_val: upper boundary
        :type max_val: float, default=None
        '''
        generator = MaskGenerator(mode='threshold', threshold_value=(min_val, max_val))
        return generator

    @staticmethod
    def quantile(min_quantile=None, max_quantile=None):
        r'''This creates a MaskGenerator which generates a mask for values outside boundaries

        :param min_quantile: lower quantile, 0-1
        :type min_quantile: float, default=None
        :param max_quantile: upper quantile, 0-1
        :type max_quantile: float, default=None
        '''
        generator = MaskGenerator(mode='quantile', quantiles=(min_quantile, max_quantile))
        return generator

    @staticmethod
    def mean(axis=None, threshold_factor=3, window=None):
        r'''This creates a MaskGenerator which generates a mask for values outside a multiple of standard-deviations from the mean.

        abs(A - mean(A)) > threshold_factor * std(A).

        :param threshold_factor: scale factor of standard-deviations to use as threshold
        :type threshold_factor: float, default=3
        :param axis: specify axis (int) to calculate mean. If no axis is specified then operates over flattened array.
        :type axis: int
        :param window: specify number of pixels to use in calculation of a rolling mean
        :type window: int, default=None
        '''
        if window is None:
            generator = MaskGenerator(mode='mean', threshold_factor=threshold_factor, axis=axis)
        else:
            generator = MaskGenerator(mode='movmean', threshold_factor=threshold_factor, axis=axis, window=window)

        return generator

    @staticmethod
    def median(axis=None, threshold_factor=3, window=None):
        r'''This creates a MaskGenerator which generates a mask for values outside a multiple of median absolute deviation (MAD) from the median.

        abs(A - median(A)) > threshold_factor * MAD(A),
        MAD = c*median(abs(A-median(A))) where c=-1/(sqrt(2)*erfcinv(3/2))

        :param threshold_factor: scale factor of MAD to use as threshold
        :type threshold_factor: float, default=3
        :param axis: specify axis (int) to calculate median. If no axis is specified then operates over flattened array.
        :type axis: int
        :param window: specify number of pixels to use in calculation of a rolling median
        :type window: int, default=None
        '''

        if window is None:
            generator = MaskGenerator(mode='median', threshold_factor=threshold_factor, axis=axis)
        else:
            generator = MaskGenerator(mode='movmedian', threshold_factor=threshold_factor, axis=axis, window=window)

        return generator

    def __init__(self,
                 mode='special_values',
                 threshold_value=(None, None),
                 quantiles=(None, None),
                 threshold_factor=3,
                 window=5,
                 axis=None):
        r'''Detects outliers in a numpy array and returns a boolean mask, True where outliers were detected.

            :param mode: a method for detecting outliers (special_values, nan, inf, threshold, quantile, mean, median, movmean, movmedian)
            :type mode: string, default='special_values'
            :param threshold_value: specify lower and upper boundaries if 'threshold' mode is selected
            :type threshold_value: tuple
            :param quantiles: specify lower and upper quantiles if 'quantile' mode is selected
            :type quantiles: tuple
            :param threshold_factor: scales detection threshold (standard deviation in case of 'mean', 'movmean' and median absolute deviation in case of 'median', 'movmedian')
            :type threshold_factor: float, default=3
            :param window: specify running window if 'movmean' or 'movmedian' mode is selected
            :type window: int, default=5
            :param axis: specify axis (int) to calculate statistics for 'mean', 'median', 'movmean', 'movmedian' modes
            :type axis: int
            :return: returns a boolean numpy array, True where outliers were detected
            :rtype: numpy.ndarray

        - special_values    test element-wise for both inf and nan
        - nan               test element-wise for nan
        - inf               test element-wise for inf
        - threshold         test element-wise if array values are outside boundaries
                            given by threshold_values = (float,float).
                            You can specify only lower threshold value by setting the other to None
                            such as threshold_values = (float,None), then
                            upper boundary will be amax(data). Similarly, to specify only upper
                            boundary, use threshold_values = (None,float).
        - quantile          test element-wise if array values are outside boundaries
                            given by quantiles = (q1,q2), 0<=q1,q2<=1.
                            You can specify only lower quantile value by setting the other to None
                            such as quantiles = (float,None), then
                            upper boundary will be amax(data). Similarly, to specify only upper
                            boundary, use quantiles = (None,float).
        - mean              test element-wise if
                            abs(A - mean(A)) > threshold_factor * std(A).
                            Default value of threshold_factor is 3. If no axis is specified,
                            then operates over flattened array. Alternatively operates along the
                            specified axis.
        - median            test element-wise if
                            abs(A - median(A)) > threshold_factor * scaled MAD(A),
                            scaled median absolute deviation (MAD) is defined as
                            c*median(abs(A-median(A))) where c=-1/(sqrt(2)*erfcinv(3/2))
                            Default value of threshold_factor is 3. If no axis is specified,
                            then operates over flattened array. Alternatively operates along the
                            specified axis.
        - movmean           the same as mean but uses rolling mean with a specified window,
                            default window value is 5
        - movmedian         the same as median but uses rolling median with a specified window,
                            default window value is 5

        '''

        self.mode = mode
        self.threshold_value = threshold_value
        self.threshold_factor = threshold_factor
        self.quantiles = quantiles
        self.window = window
        self.axis = axis

    def check_input(self, arr):

        if self.mode not in ['special_values', 'nan', 'inf', 'threshold', 'quantile',
                             'mean', 'median', 'movmean', 'movmedian']:
            raise ValueError("Wrong mode. One of the following is expected:\n" +
                            "special_values, nan, inf, threshold, \n quantile, mean, median, movmean, movmedian")

        if self.axis is not None:
            if not isinstance(self.axis, int):
                raise ValueError("axis must be an int, got {}".format(type(self.axis)))
            if self.axis < 0 or self.axis >= arr.ndim:
                raise ValueError("axis {} is out of bounds for array of dimension {}".format(self.axis, arr.ndim))

        return True

    def process(self, data):
        r'''Generate the outlier mask for a numpy array.

        :param data: input array
        :type data: numpy.ndarray
        :return: boolean array, True where outliers were detected
        :rtype: numpy.ndarray
        '''

        arr = numpy.asarray(data)

        self.check_input(arr)

        ndim = arr.ndim
        axis_index = self.axis

        # initialise mask with all False (no outliers)
        mask = numpy.zeros(arr.shape, dtype=bool)

        if self.mode == 'special_values':

            mask[numpy.logical_or(numpy.isnan(arr), numpy.isinf(arr))] = True

        elif self.mode == 'nan':

            mask[numpy.isnan(arr)] = True

        elif self.mode == 'inf':

            mask[numpy.isinf(arr)] = True

        elif self.mode == 'threshold':

            if not isinstance(self.threshold_value, tuple):
                raise ValueError("Threshold value must be given as a tuple containing two values,\n" +
                    "use None if no threshold value is given")

            threshold = self._parse_threshold_value(arr, quantile=False)

            mask[numpy.logical_or(arr < threshold[0], arr > threshold[1])] = True

        elif self.mode == 'quantile':

            if not isinstance(self.quantiles, tuple):
                raise ValueError("Quantiles must be given as a tuple containing two values,\n " +
                    "use None if no quantile value is given")

            quantile = self._parse_threshold_value(arr, quantile=True)

            mask[numpy.logical_or(arr < quantile[0], arr > quantile[1])] = True

        elif self.mode == 'mean':

            if axis_index is not None:
                slice_obj = [slice(None)] * ndim
                slice_obj[axis_index] = numpy.newaxis
                slice_obj = tuple(slice_obj)

                mean_array = numpy.mean(arr, axis=axis_index)[slice_obj]
                std_array = numpy.std(arr, axis=axis_index)[slice_obj]
                mask[numpy.abs(arr - mean_array) > self.threshold_factor * std_array] = True

            else:
                mask[numpy.abs(arr - numpy.mean(arr)) > self.threshold_factor * numpy.std(arr)] = True

        elif self.mode == 'median':

            c = -1 / (numpy.sqrt(2) * special.erfcinv(3 / 2))

            if axis_index is not None:
                slice_obj = [slice(None)] * ndim
                slice_obj[axis_index] = numpy.newaxis
                slice_obj = tuple(slice_obj)

                tmp = numpy.abs(arr - numpy.median(arr, axis=axis_index)[slice_obj])
                median_absolute_dev = numpy.median(tmp, axis=axis_index)[slice_obj]
                mask[tmp > (self.threshold_factor * c) * median_absolute_dev] = True

            else:
                tmp = numpy.abs(arr - numpy.median(arr))
                mask[tmp > (self.threshold_factor * c) * numpy.median(tmp)] = True

        elif self.mode == 'movmean':

            if axis_index is not None:
                kernel = [1] * ndim
                kernel[axis_index] = self.window
                kernel = tuple(kernel)

                mean_array = ndimage.generic_filter(arr, numpy.mean, size=kernel, mode='reflect')
                std_array = ndimage.generic_filter(arr, numpy.std, size=kernel, mode='reflect')

                mask[numpy.abs(arr - mean_array) > self.threshold_factor * std_array] = True

            else:
                mean_array = ndimage.generic_filter(arr, numpy.mean, size=(self.window,) * ndim, mode='reflect')
                std_array = ndimage.generic_filter(arr, numpy.std, size=(self.window,) * ndim, mode='reflect')

                mask[numpy.abs(arr - mean_array) > self.threshold_factor * std_array] = True

        elif self.mode == 'movmedian':

            c = -1 / (numpy.sqrt(2) * special.erfcinv(3 / 2))

            if axis_index is not None:
                kernel_shape = [1] * ndim
                kernel_shape[axis_index] = self.window
                kernel_shape = tuple(kernel_shape)

                median_array = ndimage.median_filter(arr, size=kernel_shape, mode='reflect')

                tmp = numpy.abs(arr - median_array)
                mask[tmp > (self.threshold_factor * c) * ndimage.median_filter(tmp, size=kernel_shape, mode='reflect')] = True

            else:
                kernel_shape = tuple([self.window] * ndim)
                median_array = ndimage.median_filter(arr, size=kernel_shape, mode='reflect')

                tmp = numpy.abs(arr - median_array)
                mask[tmp > (self.threshold_factor * c) * ndimage.median_filter(tmp, size=kernel_shape, mode='reflect')] = True

        else:
            raise ValueError('Mode not recognised. One of the following is expected: ' +
                              'special_values, nan, inf, threshold, quantile, mean, median, movmean, movmedian')

        return mask

    def _parse_threshold_value(self, arr, quantile=False):

        lower_val = None
        upper_val = None

        if quantile:
            if self.quantiles[0] is not None:
                lower_val = numpy.quantile(arr, self.quantiles[0])
            if self.quantiles[1] is not None:
                upper_val = numpy.quantile(arr, self.quantiles[1])
        else:
            if self.threshold_value[0] is not None:
                lower_val = self.threshold_value[0]
            if self.threshold_value[1] is not None:
                upper_val = self.threshold_value[1]

        if lower_val is None:
            lower_val = numpy.amin(arr)

        if upper_val is None:
            upper_val = numpy.amax(arr)

        if upper_val <= lower_val:
            raise ValueError("Upper threshold value must be larger than " +
                "lower threshold value or min of data")

        return (lower_val, upper_val)
