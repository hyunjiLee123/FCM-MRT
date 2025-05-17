import random
import numpy as np
from PIL import Image


class FreqTune_zhonly(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32
        img = np.array(x).astype(np.uint8)
        fft_1 = np.fft.fftn(img)

        # 랜덤 영역 뽑기
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)

        Zh_matrix = fft_1[x_min:x_max, y_min:y_max]

        # Ch 만들기
        A = 5
        a = np.random.uniform(0, A)
        Ch_matrix = np.random.uniform(-a, a, size=Zh_matrix.shape)
        # 행렬곱, 다시 넣기
        fft_1[x_min:x_max, y_min:y_max] = Zh_matrix * Ch_matrix

        img = np.fft.ifftn(fft_1)
        new_image = np.clip(img, 0, 255).astype(np.uint8)

        x = Image.fromarray(new_image)
        # x.show()
        return x

class baseline(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x
        x.show()
        return x

class OriginalFreqTune(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32
        img = np.array(x).astype(np.uint8)
        fft_1 = np.fft.fftn(img)

        # img pixel: matrix, make array: array
        # 랜덤 영역 뽑기
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)
        # 중심 좌표 구하기
        matrix = fft_1[x_min:x_max, y_min:y_max]

        # 강도
        B = 0.5
        b = np.random.uniform(0, B)
        array2 = np.random.uniform(1-b, 1+b, size=fft_1.shape)

        # array2_m = np.random.uniform(-1-b, -1+b, size=fft_1.shape)
        # array2_p = np.random.uniform(1 - b, 1 + b, size=fft_1.shape)
        # array2 = np.where(np.random.rand(*fft_1.shape) > 0.5, array2_p, array2_m)

        A = 5
        a = np.random.uniform(0, A)
        array1 = np.random.uniform(-a, a, size=matrix.shape)

        # 행렬곱, 다시 넣기
        fft_1 = fft_1 * array2
        # 행렬곱, 다시 넣기
        fft_1[x_min:x_max, y_min:y_max] = matrix * array1

        img = np.fft.ifftn(fft_1)

        # img = img.astype(np.uint8)
        # x = Image.fromarray(img)
        new_image = np.clip(img, 0, 255).astype(np.uint8)
        x = Image.fromarray(new_image)
        # x.show()
        return x


#GPT
class ImprovedFreqTune:
    def __init__(self, probability=0.5, A=5, B=0.5):
        self.probability = probability
        self.A = A
        self.B = B

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        # 이미지를 배열로 변환
        img = np.array(x).astype(np.uint8)
        height, width = img.shape[:2]

        # FFT 수행 (shift 없이)
        fft_1 = np.fft.fftn(img)

        # 랜덤으로 고주파수 영역 선택
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)

        # 고주파수 영역 강한 변형 적용
        high_freq_region = fft_1[y_min:y_max, x_min:x_max]
        a = np.random.uniform(0, self.A)
        amplitude_perturb = np.random.uniform(-a, a, size=high_freq_region.shape)
        fft_1[y_min:y_max, x_min:x_max] = high_freq_region * (1 + amplitude_perturb)

        # 저주파수 영역 약한 변형 적용
        b = np.random.uniform(0, self.B)
        global_perturb = np.random.uniform(1 - b, 1 + b, size=fft_1.shape)
        fft_1 *= global_perturb

        # 역 FFT로 변환
        img_ifft = np.fft.ifftn(fft_1)
        new_image = np.clip(np.abs(img_ifft), 0, 255).astype(np.uint8)

        # PIL 이미지로 변환
        return Image.fromarray(new_image)


class FreqTune(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32
        img = np.array(x).astype(np.uint8)
        fft_1 = np.fft.fftn(img)

        # img pixel: matrix, make array: array
        # 랜덤 영역 뽑기
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)
        # 중심 좌표 구하기
        center = ((x_max-x_min) // 2, (y_max-y_min) // 2)
        matrix = fft_1[x_min:x_max, y_min:y_max]

        array1 = np.zeros(matrix.shape)
        array2 = np.zeros(fft_1.shape)

        corners_1 = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]
        corners_2 = [(0, 0), (0, 31), (31, 0), (31, 31)]
        max_distances_2 = [np.sqrt((center[0] - corner[0]) ** 2 + (center[1] - corner[1]) ** 2) for corner in corners_2]
        max_distance_2 = max(max_distances_2)

        sigma = 10
        gamma = 0.45
        # 전체를 위 center 기준으로 퍼트리기
        # 강도
        B = 0.5
        for i in range(32):  # x축 크기
            for j in range(32):  # y축 크기
                # 중심으로부터의 거리 계산
                distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                # line
                value = max_distance_2 - distance
                # 선형 감소한 값을 거리로 정규화
                normalized_value = B * (value / max_distance_2)**2
                # normalized_value = B / distance

                # 배열에 값 저장
                array2[i, j] = normalized_value

                # # Gaussian
                # G_value = B * np.exp(-((distance/max_distance_2)**2) / (2 * (sigma ** 2)))
                # array2[i, j] = G_value

        # 행렬곱, 다시 넣기
        fft_1 = fft_1 * array2

        # Ch 만들기
        max_distances_1 = [np.sqrt((center[0] - corner[0]) ** 2 + (center[1] - corner[1]) ** 2) for corner in corners_1]
        max_distance_1 = max(max_distances_1)
        # max_log_1 = np.log(max_distance_1/epsilon)
        A = 5
        for i in range(x_max-x_min):  # x축 크기
            for j in range(y_max-y_min):  # y축 크기
                # 중심으로부터의 거리 계산
                distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                # line
                value = max_distance_1 - distance
                # 선형 감소한 값을 거리로 정규화
                normalized_value = A * (value / max_distance_1)**2
                # 배열에 값 저장
                array1[i, j] = normalized_value

                # # Gaussian
                # G_value = A * np.exp(-((distance/max_distance_1)**2) / (2 * (sigma ** 2)))
                # array1[i, j] = G_value

        # 행렬곱, 다시 넣기
        fft_1[x_min:x_max, y_min:y_max] = matrix * array1

        img = np.fft.ifftn(fft_1)

        # img = img.astype(np.uint8)
        # x = Image.fromarray(img)
        new_image = np.clip(img, 0, 255).astype(np.uint8)
        x = Image.fromarray(new_image)
        # x.show()
        return x

class TwoRegionFreqTune(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32
        img = np.array(x)
        fft_1 = np.fft.fftn(img)

        # 1번 랜덤 영역 뽑기
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)

        # 2번 랜덤 영역 뽑기
        x_min_2 = np.random.randint(width // 32, x_min+1)
        x_max_2 = np.random.randint(x_max, width - width // 32)
        y_min_2 = np.random.randint(height // 32, y_min+1)
        y_max_2 = np.random.randint(y_max, height - height // 32)

        # 랜덤 배열 만들기
        matrix_1 = fft_1[x_min:x_max, y_min:y_max]
        matrix_2 = fft_1[x_min_2:x_max_2, y_min_2:y_max_2]
        line_array = np.zeros((x_max_2-x_min_2, y_max_2-y_min_2))

        # 3번 배열
        B = 0.5
        b = np.random.uniform(0, B)
        array3 = np.random.uniform(1 - b, 1 + b, size=fft_1.shape)
        # # 3번 배열 행렬곱, 다시 넣기
        # fft_1 = fft_1 * array3

        # 1번 배열
        A = 5
        a = np.random.uniform(0, A)
        array1 = np.random.uniform(-a, a, size=matrix_1.shape)
        # # 1번 배열 행렬곱, 다시 넣기
        # fft_1[x_min:x_max, y_min:y_max] = matrix_1 * array1

        # # 중심 좌표
        # center = ((x_max-x_min) // 2, (y_max-y_min) // 2)
        #
        # corners_1 = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]
        # corners_2 = [(x_min_2, y_min_2), (x_min_2, y_max_2), (x_max_2, y_min_2), (x_max_2, y_max_2)]
        # corners_3 = [(0, 0), (0, 31), (31, 0), (31, 31)]
        #
        # max_distances_2 = [np.sqrt((center[0] - corner[0]) ** 2 + (center[1] - corner[1]) ** 2) for corner in corners_2]
        # max_distance_2 = max(max_distances_2)

        # 3차원 배열의 중심 좌표 -> 고주파수의 중심
        center_index = tuple(s // 2 for s in matrix_1.shape)
        # center = matrix_2[center_index]

        # 각 좌표의 거리 계산
        x_index, y_index, z_index = center_index
        x = np.arange(matrix_2.shape[0])  # x 축 좌표 (0 ~ shape[0]-1)
        y = np.arange(matrix_2.shape[1])  # y 축 좌표 (0 ~ shape[1]-1)
        z = np.arange(matrix_2.shape[2])  # z 축 좌표 (0 ~ shape[2]-1)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        distances = np.sqrt(
            (xx - center_index[0]) ** 2 +
            (yy - center_index[1]) ** 2 +
            (zz - center_index[2]) ** 2
        )

        # 2번 배열
        # C = 2.5
        # c = np.random.uniform(0, C
        # )
        c_down = np.random.uniform(-a, 1-b)
        c_up = np.random.uniform(1+b, a)
        # uniform
        array2 = np.random.uniform(c_down, c_up, size=matrix_2.shape)
        # array2 = np.where(np.random.rand(*matrix_2.shape) > 0.5, c_down, c_up)

        ##### 곡선/선형 변화 ########
        # 거리를 최대 0부터 배열의 대각선 길이까지 정규화
        # max_distance = np.sqrt(sum((np.array(matrix_2.shape) // 2) ** 2))
        # normalized_distances = distances / max_distance
        # # # 선형적으로 값을 변환(test6)
        # # array2 = c_up - (c_up - c_down) * normalized_distances
        # # 곡선으로 값을 변환(test7)
        # k = 10
        # array2 = c_down + (c_up - c_down) * (1 - np.log(1 + k * normalized_distances) / np.log(1 + k))
        #####################

        #######
        # 3번 배열 행렬곱, 다시 넣기
        fft_1 = fft_1 * array3
        # 2번 배열 행렬곱, 다시 넣기
        fft_1[x_min_2:x_max_2, y_min_2:y_max_2] = matrix_2 * array2
        # 1번 배열 행렬곱, 다시 넣기
        fft_1[x_min:x_max, y_min:y_max] = matrix_1 * array1
        #######

        img = np.fft.ifftn(fft_1)
        new_image = np.clip(img, 0, 255).astype(np.uint8)
        x = Image.fromarray(new_image)
        x.show()
        return x

class BlockFreqTune(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x
        ##
        ####nxn
        aa = [0, 8, 16, 24]
        bb = [0, 8, 16, 24]
        cc = [8, 16, 24, 32]
        dd = [8, 16, 24, 32]
        n = 4
        image_list = []
        for i in range(n):
            for j in range(n):
                imgcrop = x.crop((aa[i], bb[j], cc[i], dd[j]))
                image_list.append(imgcrop)
        ####

        result_image_list = []
        ##
        for img in image_list:
            length = 8
            img = np.array(img).astype(np.uint8)
            fft_1 = np.fft.fftn(img)

            # 랜덤 영역 뽑기
            x_min = np.random.randint(length // length, length // 2)
            x_max = np.random.randint(length // 2, length - length // length)
            y_min = np.random.randint(length // length, length // 2)
            y_max = np.random.randint(length // 2, length - length // length)

            Zh_matrix = fft_1[x_min:x_max, y_min:y_max]

            # 전체를 zl로 생각
            # Cl 만들기
            B = 0.5
            b = np.random.uniform(0, B)
            cl_matrix = np.random.uniform(1 - b, 1 + b, size=fft_1.shape)
            # 행렬곱, 다시 넣기
            fft_1 = fft_1 * cl_matrix

            # Ch 만들기
            A = 5
            a = np.random.uniform(0, A)
            Ch_matrix = np.random.uniform(-a, a, size=Zh_matrix.shape)
            # 행렬곱, 다시 넣기
            fft_1[x_min:x_max, y_min:y_max] = Zh_matrix * Ch_matrix

            newimg = np.fft.ifftn(fft_1)

            newimg = np.clip(newimg.real, 0, 255).astype(np.uint8)
            result_image_list.append(newimg)

        ####nxn
        row = []
        for ii in range(n):
            r = np.concatenate(result_image_list[ii * n : (ii + 1) * n], axis=0)
            row.append(r)
        new_image = np.concatenate(row, axis=1)
        ####

        x = Image.fromarray(new_image)
        # x.show()

        return x

class RangeChangeFreqTune(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32
        img = np.array(x).astype(np.uint8)
        fft_1 = np.fft.fftn(img)

        # img pixel: matrix, make array: array
        # 랜덤 영역 뽑기
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)
        # 중심 좌표 구하기
        matrix = fft_1[x_min:x_max, y_min:y_max]

        # 강도
        B = 0.5
        # b = np.random.uniform(0, B)
        # array2 = np.random.uniform(1-b, 1+b, size=fft_1.shape)
        array2 = np.random.uniform(1-B, 1+B, size=fft_1.shape)        # one uniform
        # 행렬곱, 다시 넣기
        fft_1 = fft_1 * array2

        A = 5
        # a = np.random.uniform(0, A)
        # array1 = np.random.uniform(-a, a, size=matrix.shape)
        # array1 = np.random.uniform(-A, A, size=matrix.shape)          # one uniform
        # 고주파수 범위 제외
        lower_part = np.random.uniform(-A, 1 - B, size=matrix.size // 2 + matrix.size % 2)
        upper_part = np.random.uniform(1 + B, A, size=matrix.size // 2)
        array1 = np.concatenate((lower_part, upper_part))
        np.random.shuffle(array1)
        array1 = array1.reshape(matrix.shape)
        # 행렬곱, 다시 넣기
        fft_1[x_min:x_max, y_min:y_max] = matrix * array1

        img = np.fft.ifftn(fft_1)

        # img = img.astype(np.uint8)
        # x = Image.fromarray(img)
        new_image = np.clip(img, 0, 255).astype(np.uint8)
        x = Image.fromarray(new_image)
        # x.show()
        return x


class NormalFreqtune(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32
        img = np.array(x).astype(np.uint8)
        fft_1 = np.fft.fftn(img)

        # img pixel: matrix, make array: array
        # 랜덤 영역 뽑기
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)
        # 중심 좌표 구하기
        matrix = fft_1[x_min:x_max, y_min:y_max]

        # sigma = 1
        # B = 0.5
        # array2 = np.random.normal(1, sigma, size=fft_1.shape) * B       ## ex1, ex2
        # array2 = np.clip(array2, 1-B, 1+B)                                  ## ex2
        #################

        sigma2 = 0.167                                                        # ex3(B=0.167), ex5(sigma2=0.167), ex6(array2에서 * B 제거)
        array2 = np.random.normal(1, sigma2, size=fft_1.shape)           ## ex3(all erase)

        # B = 0.5
        # array2 = np.random.normal(1, sigma, size=fft_1.shape)
        # array2 = np.clip(array2, 1-B, 1+B)                                ## ex4(all erase)

        # 행렬곱, 다시 넣기
        fft_1 = fft_1 * array2

        ##############
        # A = 5
        # array1 = np.random.normal(0, sigma, size=matrix.shape) * A      ## ex1, ex2
        # array1 = np.clip(array1, -A, A)                                     ## ex2
        #############

        sigma1 = 1.67                                                           # ex3(A=1.67), ex5(sigma1=1.67), ex6(array2에서 * A 제거)
        array1 = np.random.normal(0, sigma1, size=matrix.shape)             ## ex3(all erase)

        # A = 5
        # array1 = np.random.normal(0, sigma, size=matrix.shape)
        # array1 = np.clip(array1, -A, A)                                           ## ex4(all erase)

        # 행렬곱, 다시 넣기
        fft_1[x_min:x_max, y_min:y_max] = matrix * array1

        img = np.fft.ifftn(fft_1)

        # img = img.astype(np.uint8)
        # x = Image.fromarray(img)
        new_image = np.clip(img, 0, 255).astype(np.uint8)
        x = Image.fromarray(new_image)
        # x.show()
        return x