#
#   Author: Monica Amaya, Savannah McCoy, Seedy Jahateh, Isaac Gray, Deanna M. Wilborne
#  Created: 2023-07-06
# Location: Berea College, Berea, Kentucky
#   Course: 2023 URCPP (Undergraduate Research and Creative Projects Program)
#     Term:
#  Purpose: A set of classes that provide a rapid method of counting objects in relatively small astro-photographs
#
# History:
#           2023-07-06, DMW, created from composite of code developed during research
#
#   Notes:
#           Copyright (c) 2023, Monica Amaya, Savannah McCoy, Seedy Jahahteh, Isaac Gray, Deanna M. Wilborne
#           MIT License, see: https://github.com/mewiii/2023-URCPP-Astro-Object-Detector/blob/main/LICENSE

from matplotlib import image
import datetime as dt
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation


class GetImageFile:

    def __init__(self, image_file_name: str = "") -> None:
        self.image_raw = None  # the original unprocessed data
        self.image_file_name = image_file_name
        self.timings = {}
        try:
            start_time = dt.datetime.now()
            if image_file_name != "":
                self.image_raw = image.imread(image_file_name)
                self.image_rows = self.image_height = self.image_raw.shape[0]
                self.image_cols = self.image_width = self.image_raw.shape[1]
                self.image_dims = self.image_raw.shape[2]
            self.save_timing("IMAGE_LOAD", start_time)

            # make a 3 color slice from a 4+ dimensional object
            start_time = dt.datetime.now()
            if self.image_dims > 3:
                # self.image_3dim = np.delete(self.image_data, -1, axis=2)
                self.image_data = self.image_raw[:, :, :-1]
            else:
                self.image_data = self.image_raw.copy()
            self.save_timing("3DIM", start_time)

        except FileNotFoundError:
            print("{} not found.".format(image_file_name))

    def save_timing(self, timing_key: str, start_time) -> None:
        elapsed_time = dt.datetime.now() - start_time
        self.timings[timing_key] = [str(elapsed_time), elapsed_time]

    def show_timings(self):
        for key in self.timings.keys():
            print("{}: {}".format(key, self.timings[key][0]))


class ImageBase(GetImageFile):

    def __init__(self, image_file_name: str) -> None:
        super().__init__(image_file_name)
        self.color_tagged_image = None
        self.image_rows = None
        self.image_cols = None
        self.image_colors = None
        self.image_color_tagged = None
        self.image_rectangles = None

        if self.image_data is not None:
            self.image_rows, self.image_cols, self.image_colors = self.image_data.shape

    def color_tag_coord(self, coordinates: [] = None, color: [] = None, pastels=False) -> None:
        def random_pastel() -> float:
            return random.random() * 0.5 + 0.5

        def random_triple_pastel() -> (float, float, float):
            return random_pastel(), random_pastel(), random_pastel()

        if coordinates is None:
            # coordinates = []
            return  # short circuit, nothing to be done
        start_time = dt.datetime.now()
        if color is None:
            color = [1.0, 0.0, 0.0]  # red
        self.image_color_tagged = self.image_data.copy()
        for coord_pair in coordinates:
            r, c = coord_pair[1]
            if pastels:
                self.image_color_tagged[r][c] = random_triple_pastel()
            else:
                self.image_color_tagged[r][c] = color
        self.save_timing("COLOR_TAG", start_time)

    def draw_rectangles(self, object_list, color=(0.0, 1.0, 1.0)) -> None:  # default color cyan
        start_time = dt.datetime.now()
        image_rectangles = self.image_data.copy()
        for o in object_list:
            upper_left, lower_right = o[0], o[1]
            for r in range(upper_left[0], lower_right[0] + 1):
                image_rectangles[r][upper_left[1]] = color
                image_rectangles[r][lower_right[1]] = color
            for c in range(upper_left[1]+1, lower_right[1]):
                image_rectangles[upper_left[0]][c] = color
                image_rectangles[lower_right[0]][c] = color
        self.image_rectangles = image_rectangles.copy()
        self.save_timing("DRAW_RECTANGLES", start_time)
        return image_rectangles

    def show_image(self, image_data=None):
        if image_data is None:
            image_data = self.image_data
        plt.imshow(image_data)
        plt.show()

    # 2023-06-04, DMW, added this method
    def save_image(self, image_file_name, image_data=None):
        if image_data is None:
            image_data = self.image_data
        plt.imshow(image_data)
        plt.savefig(image_file_name)

    def save_timing(self, timing_key: str, start_time) -> None:
        elapsed_time = dt.datetime.now() - start_time
        self.timings[timing_key] = [str(elapsed_time), elapsed_time]


class CourseObjectDetector(ImageBase):

    def __init__(self, image_file_name: str = "") -> None:
        super().__init__(image_file_name)
        self.test_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        self.course_objects_list = []

        if self.image_data is not None:
            self.find_objects()

    def find_objects(self) -> []:
        start_time = dt.datetime.now()
        objects = []
        for r in range(self.image_rows):
            for c in range(self.image_cols):
                test_pixel_brightness = self.image_data[r][c].mean()
                found_object = True

                for d in self.test_offsets:
                    test_r = r + d[0]  # add the row direction offset to the current r
                    test_c = c + d[1]  # add the col direction offset to the current c
                    if test_r < 0 or test_r >= self.image_rows:
                        continue  # skip invalid rows
                    if test_c < 0 or test_c >= self.image_cols:
                        continue  # skip invalid columns
                    border_pixel_brightness = self.image_data[test_r][test_c].mean()
                    if border_pixel_brightness > test_pixel_brightness:
                        found_object = False
                        break  # brighter object found, skip to next pixel

                if found_object:
                    objects.append([test_pixel_brightness, (r, c)])

        objects.sort(reverse=True)
        self.course_objects_list = objects
        elapsed_time = dt.datetime.now() - start_time
        self.timings["COURSE_COUNT"] = [str(elapsed_time), elapsed_time]
        return objects


class RefinedObjectDetector(CourseObjectDetector):

    def __init__(self, image_file_name: str = "", above_multi=1.0) -> None:
        super().__init__(image_file_name)
        self.refined_objects_list = []
        self.image_mean = self.image_data.mean()  # TODO: 2023-06-01, DMW, need to generalize this

        # if we have a list of course objects, refine the objects detected
        if len(self.course_objects_list) > 0:
            self.refine_detected_objects(above_multi * self.image_mean)

    def refine_detected_objects(self, above_value: float) -> None:
        self.find_objects_at_or_above_value(self.course_objects_list, above_value)

    # given a list of object coordinates, find pixels above a specified value
    def find_objects_at_or_above_value(self, objects, above_value):
        start_time = dt.datetime.now()
        output_objects = objects.copy()
        for o in objects:
            r = o[1][0]
            c = o[1][1]

            if o[0] < above_value:
                output_objects.remove(o)

        self.refined_objects_list = output_objects
        elapsed_time = dt.datetime.now() - start_time
        self.timings["REFINED_COUNT"] = [str(elapsed_time), elapsed_time]
        return output_objects


# Class for computing Standard Deviation
class SD:

    # constructor takes an optional list of data
    def __init__(self, data: []) -> None:
        if len(data) == 0:
            raise ValueError("Length of data is 0.")

        self.data = data
        self.sum = sum(self.data)
        self.mean = float(self.sum) / float(len(data))
        self.max = max(data)
        self.min = min(data)
        self.sum_of_sq_diffs = 0
        self.variance = 0
        self.sigma = 0
        self.calc_sd()

    # calculate the standard deviation
    def calc_sd(self) -> None:
        for val in self.data:
            self.sum_of_sq_diffs += (val - self.mean) ** 2.0

        self.variance = self.sum_of_sq_diffs / len(self.data)
        self.sigma = self.variance ** 0.5

    # summarize the simple statistics
    def __str__(self) -> str:
        out = "     Max: {}\n".format(self.max) +\
              "    Mean: {}\n".format(self.mean) +\
              "     Min: {}\n".format(self.min) +\
              "   Sigma: {}\n".format(self.sigma) +\
              "Variance: {}".format(self.variance)
        return out

    def to_string(self) -> None:
        print(self)


# 2023-06-04, DMW, very loosely based on: https://dbader.org/blog/python-iterators
class QualitativeColorIterator:

    def __init__(self):
        self.map = 0
        self.index = [0, 0, 0]
        self.max = 20

    def __next__(self):
        if self.map == 0:
            result = plt.cm.tab20(self.index[self.map])
        elif self.map == 1:
            result = plt.cm.tab20b(self.index[self.map])
        else:
            result = plt.cm.tab20c(self.index[self.map])

        self.index[self.map] += 1
        if self.index[self.map] >= self.max:
            self.index[self.map] = 0
            self.map += 1
            if self.map > 2:
                self.map = 0

        return result


class PrecisionObjectDetector(RefinedObjectDetector):

    def __init__(self, image_file_name: str = "",
                 animate_object_detection=False,
                 above_multi=1.0,
                 debug=False,
                 # timeout=2
                 ) -> None:
        super().__init__(image_file_name, above_multi)
        if debug:
            self.color_tag_coord(self.course_objects_list, (1.0, 1.0, 0))
            self.show_image(self.image_color_tagged)
            self.color_tag_coord(self.refined_objects_list, (0.0, 1.0, 0.0))
            self.show_image(self.image_color_tagged)
        self.precision_objects_list = []
        self.image_erasable = None
        self.image_zero_below(self.image_mean)
        self.object_rectangles = []
        self.object_areas = []
        self.area_statistics = None
        self.object_count = 0

        # animation specific class global variables
        self.image_animation_frames_list = [self.image_data.copy()]
        self.image_animation_frame = self.image_data.copy()
        self.animate_object_detection = animate_object_detection
        self.animation_color = (0.0, 1.0, 1.0)
        self.qci = QualitativeColorIterator()

        self.erased_points = []
        if self.image_data is not None:
            try:
                self.objects_points = self.find_objects_points()
                self.get_objects_rectangles()  # this can't run without find_objects_points() completing
            except TimeoutError as e:  # this exception is not raised, test code is commented out
                self.objects_points = []
                print(e)

        else:
            self.objects_points = []  # hold all the points that define a single object

    def get_objects_rectangles(self):
        if len(self.objects_points) == 0:
            return

        start_time = dt.datetime.now()
        object_rectangles = []
        object_areas = []
        for obj in self.objects_points:
            brightness, coords = zip(*obj)
            rows, cols = zip(*coords)
            min_row, min_col = min(rows), min(cols)
            max_row, max_col = max(rows), max(cols)
            object_rectangles.append([
                (min_row, min_col),
                (max_row, max_col)
            ])
            area = (max_row - min_row+1) * (max_col - min_col+1)
            object_areas.append(area)

        self.object_rectangles = object_rectangles
        self.object_count = len(self.object_rectangles)  # the number of rectangles is the number of detected objects
        self.object_areas = object_areas
        self.area_statistics = SD(self.object_areas)
        self.save_timing("RECTANGLES_AREAS", start_time)

    def find_objects_points(self):
        def random_pastel():
            return random.random()*0.5 + 0.5

        def triple_pastel():
            return random_pastel(), random_pastel(), random_pastel()

        start_time = dt.datetime.now()
        objects_points = []
        for obj in self.refined_objects_list:
            if self.animate_object_detection:
                color = next(self.qci)
                self.animation_color = color[0], color[1], color[2]
            object_points = self.find_object_points(obj)
            if len(object_points) > 0:
                objects_points.append(object_points)

        elapsed_time = dt.datetime.now() - start_time
        self.timings["PRECISION_OBJECTS"] = [str(elapsed_time), elapsed_time]
        self.precision_objects_list = objects_points
        return objects_points

    def image_zero_below(self, below_value: float, color=(0.0, 0.0, 0.0)) -> None:
        self.image_erasable = self.image_data.copy()
        for r in range(self.image_rows):
            for c in range(self.image_cols):
                # TODO: 2023-05-31, DMW, this function needs to be generalized
                pixel_data = self.image_erasable[r][c].mean()
                if pixel_data < below_value:
                    self.image_erasable[r][c] = color  # set pixels below below_value to black

    def find_object_points(self, obj: [], obj_list=None) -> []:
        if obj_list is None:
            obj_list = []

        if obj[1] in self.erased_points:
            return obj_list
        else:
            self.erased_points.append(obj[1])

        obj_list.append(obj)

        obj_brightness = obj[0]
        obj_row, obj_col = obj[1]
        self.image_erasable[obj_row][obj_col] = (0.0, 0.0, 0.0)
        if self.animate_object_detection:
            self.image_animation_frame[obj_row][obj_col] = self.animation_color
            self.image_animation_frames_list.append(self.image_animation_frame.copy())
        for d in self.test_offsets:
            next_row = obj_row + d[0]
            next_col = obj_col + d[1]
            if 0 <= next_row < self.image_rows and 0 <= next_col < self.image_cols:
                next_brightness = self.image_erasable[next_row, next_col].mean()

                if self.image_mean <= next_brightness <= obj_brightness:
                    obj_list += self.find_object_points([next_brightness, (next_row, next_col)])

        return obj_list

    def create_animation_frames(self, obj):
        # set up for animation
        self.animate_object_detection = True
        self.image_animation_frames_list = [self.image_data.copy()]
        self.image_erasable = self.image_data.copy()  # reset image erasable data
        self.erased_points = []  # reset erased image points list
        object_points = self.find_object_points(obj)

        self.animate_object_detection = False

    # 2023-06-02, DMW, this code is loosely based on:
    # https://stackoverflow.com/questions/50413680/matplotlib-animate-2d-array
    def show_animation_frames(self):
        saf_image = self.image_animation_frames_list[0]
        i = 0  # index into image animation frames

        # set up matplotlib
        saf_fig = plt.figure()
        saf_ims = plt.imshow(saf_image, animated=True)

        # set up animation function
        def update_figure(*args):
            nonlocal i
            nonlocal saf_image
            nonlocal saf_ims

            i += 1
            if i < len(self.image_animation_frames_list):
                saf_image = self.image_animation_frames_list[i]

            saf_ims.set_array(saf_image)

            # note, the COMMA is required because: The animation function must return a sequence of Artist objects.
            return saf_ims,

        # set up animation function and start the animation
        saf_animation = animation.FuncAnimation(saf_fig, update_figure, blit=True, cache_frame_data=False)
        plt.show()

    # noinspection PyMethodMayBeStatic
    # noinspection SpellCheckingInspection
    def srgb_grayscale(self, color_tuple: ()) -> int:
        return int(color_tuple[0] * 0.2126 + color_tuple[1] * 0.7152 + color_tuple[2] * 0.0722)

    # 2023-06-04, DMW, created this method
    def save_animation_frames(self, frames_path: str = "") -> None:
        start_time = dt.datetime.now()
        if len(self.image_animation_frames_list) == 0:
            return

        i = 0
        for ani_image_data in self.image_animation_frames_list:
            image_file_name = frames_path + "/frame_{:06d}.png".format(i)
            self.save_image(image_file_name, ani_image_data)
            i += 1

        self.save_timing("SAVE_ANIMATION_FRAMES", start_time)


# limited testing
if __name__ == "__main__":
    def ___main_test___():
        test_filename = "./images/44x29-test-2011-05Andreo_BigDipper7k.png"
        pod = PrecisionObjectDetector(test_filename)
        print(pod.object_count)  # 33 is the expected output
        pod.show_timings()
    ___main_test___()
