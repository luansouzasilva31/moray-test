import cv2
import time
import numpy as np
from functools import wraps
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(
            f'{func.__name__}: {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def plot_images(images: list, grid: tuple = None, titles: list = None,
                cmaps: list = None, suptitle: str = ''):
    # Checking titles
    if not titles:
        titles = ['' for i in images]
    if not cmaps:
        cmaps = [None for i in images]
    if not grid:
        # # Infering grid from images list
        n_cols = np.ceil(np.sqrt(len(images))).astype(int)
        n_rows = np.ceil(len(images) / n_cols).astype(int)

        grid = (n_rows, n_cols)

    # Check sizes
    assert len(images) == len(titles) == len(cmaps), \
        "Lists of images, titles and cmaps must have the same length."

    # Creating plotting section
    fig, axis = plt.subplots(nrows=grid[0], ncols=grid[1], figsize=(22, 16),
                             sharex=True, sharey=True)

    count = 0
    if grid[0] == 1 and grid[1] == 1:
        axis.imshow(images[count], cmap=cmaps[count])
        axis.set_title(titles[count])

    elif grid[0] == 1 or grid[1] == 1:
        for i in range(max(grid[0], grid[1])):
            axis[i].imshow(images[count], cmap=cmaps[count])
            axis[i].set_title(titles[count])
            count += 1

    else:
        for i in range(grid[0]):
            for j in range(grid[1]):
                if count >= len(images):
                    fig.delaxes(axis[i, j])
                    continue
                axis[i, j].imshow(images[count], cmap=cmaps[count])
                axis[i, j].set_title(titles[count])
                count += 1

    plt.suptitle(suptitle)
    plt.show(block=True)

    return


class RowCropDetector:
    def __init__(self):
        pass

    @timeit
    def detect_crop_lines(self, image: np.ndarray, plot: bool = False):
        # Segmentation of plant rows
        custom_mask, closing = self.segment_row_crop(image, plot=False)

        # Majoritary row crop orientation
        ref_line, theta = self.orientation_inference(closing)

        # Oriented lines
        oriented_lines = self.get_oriented_lines(custom_mask,
                                                 ref_theta=theta)

        # Get clusters
        line_clusters = self.get_line_clusters(oriented_lines, image.shape[:2])

        # Filter cluster lines
        final_lines = self.filter_line_clusters(oriented_lines, line_clusters)

        if plot:
            # Draw
            ref_line_img = self.draw_rho_theta_lines(
                image, np.expand_dims(ref_line, axis=0))
            orient_lines_img = self.draw_rho_theta_lines(
                image, oriented_lines)
            clusters_img = self.draw_line_clusters(
                oriented_lines, line_clusters, bg_img=image
            )
            final_img = self.draw_rho_theta_lines(
                image, final_lines
            )

            # Plot
            plot_images(
                [image[:, :, ::-1], ref_line_img[:, :, ::-1],
                 orient_lines_img[:, :, ::-1], clusters_img[:, :, ::-1],
                 final_img[:, :, ::-1]],
                titles=['Input', 'Orientation', 'Lines', 'Clusters', 'Final'],
                cmaps=[None, None, None, None, None]
            )

        return

    @timeit
    def orientation_inference(self, row_crop_mask: np.ndarray):
        # Elementary lines structure
        skeleton = skeletonize(row_crop_mask)
        skeleton = (skeleton * 255).astype(np.uint8)
        # TODO: prune skeleton

        pseudo_lines = self.detect_initial_rows(
            skeleton, thresh_list=[150, 120, 100, 80, 50])

        median_line = np.median(pseudo_lines, axis=0)
        _, ref_theta = median_line[0]

        return median_line, ref_theta

    @timeit
    def get_oriented_lines(self, row_crop_mask: np.ndarray, ref_theta: float):
        skeleton = skeletonize(row_crop_mask, method='lee')

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        skeleton = cv2.dilate(skeleton, kernel)
        skeleton = cv2.Canny(skeleton, 0, 255)

        lines = cv2.HoughLines(skeleton, rho=1, theta=np.pi/180, threshold=50)
        f_lines = self.filter_lines_angle(lines, ref_theta, p=np.pi/360)

        return f_lines

    @timeit
    def get_line_clusters(self, lines: np.ndarray, image_shape: tuple):
        bg = np.zeros(image_shape[:2], dtype=np.uint8)
        white_lines = self.draw_rho_theta_lines(bg, lines, color=255, thick=2)

        kernel = np.ones((5, 5), np.uint8)
        white_lines = cv2.morphologyEx(white_lines, cv2.MORPH_CLOSE, kernel)
        white_lines = ((white_lines > 50) * 255).astype(np.uint8)

        cnts, _ = cv2.findContours(white_lines, 0, 1)

        clusters = list()
        for i, line in enumerate(lines):
            bg = np.zeros(image_shape[:2], dtype=np.uint8)
            img_line = self.draw_rho_theta_lines(bg, lines[i:i+1],
                                                 color=255, thick=2)
            for j, cnt in enumerate(cnts):
                bg = np.zeros(image_shape[:2], dtype=np.uint8)
                img_cnt = cv2.drawContours(bg, [cnt], -1, 255, -1)

                img_and = np.bitwise_and(img_cnt, img_line)

                if np.sum(img_and) >= np.sum(img_line) * 0.8:
                    clusters.append(j)
                    break

        return clusters

    @timeit
    def filter_line_clusters(self, lines, clusters_idx):
        final_lines = list()
        for k in np.unique(clusters_idx):
            idxs = np.where(clusters_idx == k)[0]
            k_lines = lines[idxs]

            k_rho, k_theta = np.median(k_lines, axis=0)[0]
            final_lines.append(np.array([[k_rho, k_theta]]))

        final_lines = np.array(final_lines)

        return final_lines

    @timeit
    def segment_row_crop(self, image: np.ndarray, plot: bool = False):
        b, g, r = cv2.split(image)

        custom_thresh = (np.bitwise_and(g >= b, g >= r) * 255).astype(np.uint8)
        custom_thresh_clean = self.clean_small_contours(
            custom_thresh, min_area=25)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        closing = cv2.morphologyEx(
            custom_thresh_clean, cv2.MORPH_CLOSE, kernel)

        if plot:
            plot_images(
                [image[:, :, ::-1], custom_thresh, custom_thresh_clean,
                 closing], grid=(1, 4),
                titles=['Input', 'Thresh', 'Cleaning', 'Closing'],
                suptitle='Row crop segmentation'
            )

        return custom_thresh, closing

    def detect_initial_rows(self, skeleton, thresh_list=[150]):

        for hough_thresh in thresh_list:
            lines = cv2.HoughLines(
                skeleton, rho=1, theta=np.pi/180, threshold=hough_thresh
            )
            if lines is not None:  # if you found some
                # TODO: verify confidence instead

                break

        return lines

    def extract_final_structure(self, mask: np.ndarray):
        return

    def filter_lines_angle(self, lines, ref_theta, p=np.pi/360):
        new_lines = list()
        for line in lines:
            _, angle = line[0]

            diff = abs(ref_theta - angle)
            if diff > np.pi / 2:
                diff = np.pi - diff

            if diff <= p:
                new_lines.append(line)

        return np.array(new_lines)

    def draw_line_clusters(self, lines, clusters_idx, bg_img: np.ndarray):
        img_lines = bg_img.copy()
        for k in np.unique(clusters_idx):
            idxs = np.where(clusters_idx == k)[0]

            color = (int(np.random.choice([0, 63, 127, 180, 255])), 0,
                     int(np.random.choice([0, 63, 127, 180, 255])))

            img_lines = self.draw_rho_theta_lines(
                img_lines, lines[idxs],
                color=color, thick=2)

        return img_lines

    @staticmethod
    def clean_small_contours(image, min_area: int = 25):
        new_img = image.copy()
        cnts, _ = cv2.findContours(image, 0, 1)

        for cnt in cnts:
            area = cv2.contourArea(cnt)

            if area < min_area:
                new_img = cv2.drawContours(new_img, [cnt], -1, 0, -1)

        return new_img

    @staticmethod
    def draw_rho_theta_lines(image, lines, color=(0, 0, 255), thick=2):
        new_image = image.copy()

        for rho, theta in np.squeeze(lines, axis=1):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))
            cv2.line(new_image, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

        return new_image


if __name__ == '__main__':
    row_crop_detector = RowCropDetector()

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print(f'\n>>>>>>>> {i} <<<<<<<<')
        image = cv2.imread(f'data/{i}.png')
        row_crop_detector.detect_crop_lines(image, plot=True)
