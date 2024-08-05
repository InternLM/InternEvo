import functools
import os
import pickle
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from internlm.core.context import Config
from internlm.simulator.common import MB, OUT_OF_MEM_LATENCY, WORLD_SIZE_LIST, CommOp

# import profiler.benchmark
# import scipy.interpolate
from internlm.simulator.profiler.benchmark.multi_head_attn import UnitMultiHeadAttn
from internlm.simulator.profiler.profiler import run_profile


class PolynomialModel:
    def __init__(self, degree, data, name="unknown", segments=None) -> None:
        """_summary_

        Args:
            degree (int): _description_
            data (dict): _description_
            segments (dict): _description_
        """
        self.name = name
        self.degree = 3  # 多项式的度数
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)  # 准备多项式回归模型
        self.data = pd.DataFrame(data)  # 转换为DataFrame
        if segments is None:
            segments = {"all": (0, float("inf"))}
        print(segments, flush=True)
        self.segments = OrderedDict(segments)
        self.segment_scores = {seg: {} for seg in self.segments}  # 用于存储拟合结果和评分
        self.model_fit = {
            seg: {card: None for card in self.data["World_Size"].unique()} for seg in self.segments
        }  # 存储模型
        self.see_base_value()
        self.build_model()

    def see_base_value(self):
        # 可视化数据
        plt.figure(figsize=(12, 6))
        for card in self.data["World_Size"].unique():
            subset = self.data[self.data["World_Size"] == card]
            plt.scatter(subset["Data_B"], subset["Latency_s"], label=f"{card} cards")

        plt.xlabel("Data Transferred (MB)")
        plt.ylabel("Latency (ms)")
        plt.title("Transferred Latency vs Data Transferred for Different Card Numbers")
        plt.legend()
        plt.xscale("log")
        plt.grid(True)
        plt.savefig(f"{self.name}.jpg")
        plt.show()
        print(self.data.head())

    def build_model(self):
        # 对每个分段和卡数的数据进行拟合
        plt.figure(figsize=(12, 6))
        for seg, (low, high) in self.segments.items():
            for card in self.data["World_Size"].unique():
                subset = self.data[
                    (self.data["World_Size"] == card) & (self.data["Data_B"] >= low) & (self.data["Data_B"] < high)
                ]

                # 如果该段中没有足够的数据点，则跳过
                if len(subset) < 2:
                    continue

                # 准备数据
                X = subset["Data_B"].values.reshape(-1, 1)
                y = subset["Latency_s"].values
                X_poly = self.poly_features.fit_transform(X)

                # 拟合模型
                model = LinearRegression()
                model.fit(X_poly, y)
                y_pred = model.predict(X_poly)
                self.model_fit[seg][card] = model

                # 评估模型
                score = r2_score(y, y_pred)
                self.segment_scores[seg][card] = score

                # 可视化拟合结果
                plt.scatter(X / MB, y, label=f"{card} cards")
                plt.plot(X / MB, y_pred, label=f"{card} cards Fit")

        # 绘制图表
        plt.xlabel("Data Transferred (MB)")
        plt.ylabel("Latency (ms)")
        plt.title("Segmented Polynomial Regression Fit for Different Card Numbers")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.name}_fit.jpg")
        plt.show()

    def return_segments(self, x):
        for key, value in self.segments.items():
            low, hight = value[0], value[1]
            if x >= low and x < hight:
                return key
        assert ValueError, f"predict value:{x} out of range"

    def predict(self, world_size, complexity):
        try:
            model = self.model_fit[self.return_segments(complexity)][world_size]
            X_pred = self.poly_features.fit_transform([[complexity]])
            Y_pred = model.predict(X_pred)[0]
            return Y_pred
        except Exception as e:
            print(f"e: {e}", flush=True)
            import pdb

            pdb.set_trace()


class SplineModel:
    def __init__(self):
        self._data_prefix = "data/cost_data"
        self.spline_model_list = {}
        self.data = {}
        self.load_data()
        self.build_model()

    def load_data(self):
        for cost_data_file in os.listdir(self._data_prefix):
            name, suffix = cost_data_file.split(".")
            if suffix == "pickle":
                with open(f"{self._data_prefix}/{cost_data_file}", "rb") as f:
                    self.data[name] = pickle.load(f)

    @staticmethod
    def reformat_data_to_cost_model(total_results):
        reformat_data = dict()
        for world_size in total_results.keys():
            list_data = []
            for complexity in total_results[world_size].keys():
                for value in total_results[world_size][complexity]:
                    list_data.append([value["lat"], complexity])  # p data[2][524288][0]['lat']

            # list_data.sort(key=functools.cmp_to_key(my_compare))
            data_list = list(map(list, zip(*list_data)))
            reformat_data[world_size] = {"Data_B": data_list[1], "Latency_s": data_list[0]}

        return reformat_data

    def build_model(self):
        # p data[2][524288][0]['lat']
        for cost_type, cost_data in self.data.items():
            if cost_type != CommOp.FLASH_ATTN:
                try:
                    cost_data = SplineModel.reformat_data_to_cost_model(cost_data)
                except TypeError as e:
                    print(f"e : {e}", flush=True)
                    import pdb

                    pdb.set_trace()

                self.spline_model_list[cost_type] = {}
                for world_size, data in cost_data.items():
                    try:
                        x = data["Data_B"]
                        y = data["Latency_s"]
                    except KeyError as e:
                        print(f"e : {e}", flush=True)
                        import pdb

                        pdb.set_trace()
                    self.spline_model_list[cost_type][world_size] = interp1d(x, y, kind="slinear")
                    # self.see_base_value(cost_type, world_size, x, y)
            else:  # fa我们直接查表，不预测
                self.spline_model_list[cost_type] = {}
                self.spline_model_list[cost_type][1] = cost_data[1]

    def predict(self, cost_type, world_size, complexity):
        return self.spline_model_list[cost_type][world_size](complexity)

    def predict_cost(self, cost_type: CommOp, complexity=0, world_size=1, **kwargs):
        """predict computation cost
        The cost of attention will use KV mapping, and the cost of linear will
        use PolynomialModel.

        Args:
            cost_type (CommOp): _description_
            complexity (int, optional): _description_. Defaults to 0.

        Returns:
            float: op latency.
        """
        if cost_type == CommOp.FLASH_ATTN:
            try:
                key = UnitMultiHeadAttn.gen_store_key(**kwargs)
                return self.spline_model_list[cost_type][1][key][0]["lat"]
            except KeyError as e:
                raise KeyError(f"not found FA key: {key}")
        else:
            try:
                if cost_type != CommOp.LINEAR and world_size == 1:
                    return 0
                else:
                    spline_model = self.spline_model_list[cost_type][world_size]
                    predict = spline_model(complexity)
            except ValueError:
                below_bounds, above_bounds = spline_model.x[0], spline_model.x[-1]
                if complexity < below_bounds:
                    return spline_model(below_bounds)  # 如果超过下界就返回下界
                if complexity > above_bounds:
                    lat = spline_model(above_bounds)
                    return lat * complexity / above_bounds  # 如果超过上界就线性扩展
                raise ValueError(f"value error for cost_type:{cost_type}, complexity:{complexity}")
            except KeyError as e:
                print(f"e : {e}", flush=True)
                import pdb

                pdb.set_trace()
            else:
                return predict


def my_compare(a, b):
    world_size_a, complexity_a = a[0], a[2]
    world_size_b, complexity_b = b[0], b[2]
    # print(world_size_a, world_size_b, complexity_a, complexity_b)

    if world_size_a > world_size_b:
        return True
    elif world_size_a < world_size_b:
        return False
    else:
        if complexity_a > complexity_b:
            return True
        elif complexity_a < complexity_b:
            return False
        else:
            assert ValueError, f"a:{a}, b:{b}"


class GenCostModel:
    def __init__(self, is_master=True, build_type_list=None) -> None:
        self._master = is_master
        self._profile_args = Config(
            {
                "trials": 10,
                "warmups": 1,
            }
        )
        self.cost_data = None
        self._data_prefix = "data/cost_data"
        self.cost_kv_data = {}
        self.build_type_list = build_type_list

    def _log(self, msg: str):
        if self._master:
            print(msg, flush=True)

    def build_cost_model_by_key_value(self):
        if self.cost_data is None:
            self.cost_data = OrderedDict()
            for bench_type in self.build_type_list:
                self._log(f"now test {bench_type}")
                self.cost_kv_data[bench_type] = run_profile(self._profile_args, bench_type)

    def load_cost_model_by_key_value(self):
        self.cost_data = OrderedDict()
        for bench_type in self.build_type_list:
            self._log(f"now load {bench_type}")
            with open(f"./data/{bench_type}.pickle", "rb") as f:
                self.cost_kv_data[bench_type] = pickle.load(f)

    def draw_pic(self, data, cost_type):
        plt.figure(figsize=(12, 6))
        world_sizes = list(data.index)
        for vol in list(data.columns):
            plt.plot(world_sizes, data[vol].values, label=f"{vol/1024**2:.2f} MB")

        plt.xlabel("GPU nums")
        plt.ylabel("Latency (s)")
        plt.title(f"{cost_type}")
        # plt.xscale("log")
        # plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./data/pics/{cost_type}.jpg")
        plt.show()

    def dump_data(self):
        # p data[2][524288][0]['lat']
        for bench_type, results in self.cost_kv_data.items():
            indexs, columns = [], None
            tables = []
            if bench_type != CommOp.FLASH_ATTN:
                for world_size, values in results.items():
                    indexs.append(world_size)
                    one_col = []
                    tmp_columns = []
                    for vol, latency in values.items():
                        tmp_columns.append(vol)
                        one_col.append(latency[0]["lat"])
                    if columns is None:
                        columns = deepcopy(tmp_columns)
                    tables.append(one_col)

                # print(f"bench_type: {bench_type}", flush=True)
                # print(f"index: {indexs}", flush=True)
                # print(f"columns: {columns}", flush=True)

                df = pd.DataFrame(tables, columns=columns, index=indexs)
                df.to_csv(f"./data/excel/{bench_type}.csv", index=False)

                if bench_type != CommOp.LINEAR:
                    self.draw_pic(df, bench_type)

            with open(f"{self._data_prefix}/{bench_type}.pickle", "wb") as f:
                pickle.dump(results, f)
