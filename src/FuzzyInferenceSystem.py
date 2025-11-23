import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class FuzzyInferenceSystem:
    """
    Python reimplementation of fuzzyR::fuzzyinferencesystem.R.
    Logical order: newfis() → addvar() → addmf() → addrule() → evalfis() + defuzz()

    当前版本只实现：
        - Mamdani 型模糊系统 (fis_type="mamdani")
        - Type-1 隶属函数 (mf_type="t1")
        - AND/OR 使用 min/max（或通过 and_method/or_method 自己改）
        - 常见 defuzz 方法：centroid, bisector, mom, som, lom

    ⚠ 关键改动：在推理路径中始终保持 torch.Tensor，不再 .item() 或转成 float，
      以便梯度可以从 defuzz 一直反传到 MF 参数。
    """

    # ======================================================================
    # Constructor (equivalent to fuzzyR::newfis)
    # ======================================================================
    def __init__(self, name, fis_type="mamdani", mf_type="t1",
                 and_method="min", or_method="max", imp_method="min",
                 agg_method="max", defuzz_method="centroid"):

        self.name = name
        self.type = fis_type          # 目前只实际支持 "mamdani"
        self.mf_type = mf_type        # 目前只实际支持 "t1"
        self.and_method = and_method
        self.or_method = or_method
        self.imp_method = imp_method
        self.agg_method = agg_method
        self.defuzz_method = defuzz_method

        self.input = []   # list of variables: {"name","range","mf":[...],...}
        self.output = []
        self.rule = []    # Mamdani 规则矩阵/列表

        print(f"Created FIS: {self.name} (type={self.type})")

    # ======================================================================
    # addvar() — equivalent to fuzzyR::addvar()
    # ======================================================================
    def add_variable(self, var_type, name, var_range,
                     method=None, params=None,
                     firing_method="tnorm.min.max"):

        variable = {
            "name": name,
            "range": list(var_range),
            "method": method,
            "params": params,
            "mf": [],                 # list of {"name","type","params"}
            "firing_method": firing_method
        }

        if var_type == "input":
            self.input.append(variable)
        elif var_type == "output":
            self.output.append(variable)
        else:
            raise ValueError("var_type must be 'input' or 'output'")

        print(f"Added {var_type}: {name} range={var_range}")

    # ======================================================================
    # addmf() — equivalent to fuzzyR::addmf()
    # ======================================================================
    def add_mf(self, var_type, var_index, mf_name, mf_type, mf_params):

        mf_data = {"name": mf_name, "type": mf_type, "params": mf_params}

        if var_type == "input":
            self.input[var_index]["mf"].append(mf_data)
        elif var_type == "output":
            self.output[var_index]["mf"].append(mf_data)
        else:
            raise ValueError("var_type must be 'input' or 'output'")

        print(f"Added MF '{mf_name}' ({mf_type}) to {var_type}[{var_index}]")

    # ======================================================================
    # addrule() — equivalent to fuzzyR::addrule()
    # ======================================================================
    def add_rule(self, rule_list):
        """
        Mamdani 规则格式（简化版，对应 fuzzyR）：
            [ante_1, ante_2, ..., ante_n, cons, weight, and_or]

        其中：
            ante_i > 0 : 使用第 ante_i 个 MF
            ante_i < 0 : 使用 NOT(MF)
            ante_i = 0 : don't care
            cons       : 输出变量使用的 MF index（正负同上）
            weight     : 规则权重（0~1）
            and_or     : 1=AND (min), 2=OR (max)
        """
        self.rule.append(rule_list)
        print(f"Added rule: {rule_list}")

    # ======================================================================
    # showrule() — equivalent to fuzzyR::showrule()
    # ======================================================================
    def show_rules(self):
        if not self.rule:
            print("No rules defined.")
            return

        print("=== Rule Base ===")
        for i, r in enumerate(self.rule, 1):
            print(f"{i}. Rule: {r}")
        print("=================")

    # ======================================================================
    # evalmf() — Standard Type-1 membership functions
    # ======================================================================
    def evalmf(self, x, mf_type, params):
        """
        计算某个 MF 在 x 处的隶属度。
        x 可以是标量或 1D tensor / list / np.array。
        返回 torch.Tensor。
        """
        # x 不需要追踪梯度（我们关心的是参数），用 as_tensor 即可
        x = torch.as_tensor(x, dtype=torch.float32)

        # 允许 params 是 list / np.array / tensor
        if isinstance(params, torch.Tensor):
            p = params
        else:
            p = torch.as_tensor(params, dtype=torch.float32)

        if mf_type == "gaussmf":
            sigma, c = p[..., 0], p[..., 1]
            return torch.exp(-((x - c) ** 2) / (2 * sigma ** 2))

        elif mf_type == "gbellmf":
            a, b, c = p[..., 0], p[..., 1], p[..., 2]
            return 1 / (1 + ((torch.abs((x - c) / a)) ** (2 * b)))

        elif mf_type == "trimf":
            a, b, c = p[..., 0], p[..., 1], p[..., 2]
            return torch.clamp(
                torch.minimum((x - a) / (b - a),
                              (c - x) / (c - b)),
                0, 1
            )

        elif mf_type == "trapmf":
            a, b, c, d = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
            y1 = (x - a) / (b - a)
            y2 = torch.ones_like(x)
            y3 = (d - x) / (d - c)
            y = torch.minimum(torch.minimum(y1, y2), y3)
            return torch.clamp(y, 0, 1)

        elif mf_type == "sigmf":
            a, c = p[..., 0], p[..., 1]
            return 1 / (1 + torch.exp(-a * (x - c)))

        elif mf_type == "smf":
            a, b = p[..., 0], p[..., 1]
            y = torch.zeros_like(x)
            idx2 = (x > a) & (x < (a + b) / 2)
            idx3 = (x >= (a + b) / 2) & (x < b)
            idx4 = x >= b
            y[idx2] = 2 * ((x[idx2] - a) / (b - a)) ** 2
            y[idx3] = 1 - 2 * ((x[idx3] - b) / (b - a)) ** 2
            y[idx4] = 1
            return y

        elif mf_type == "zmf":
            a, b = p[..., 0], p[..., 1]
            y = torch.zeros_like(x)
            idx1 = x <= a
            idx2 = (x > a) & (x < (a + b) / 2)
            idx3 = (x >= (a + b) / 2) & (x < b)
            y[idx1] = 1
            y[idx2] = 1 - 2 * ((x[idx2] - a) / (b - a)) ** 2
            y[idx3] = 2 * ((x[idx3] - b) / (b - a)) ** 2
            return y

        elif mf_type == "pimf":
            a, b, c, d = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
            smf_part = self.evalmf(x, "smf", torch.stack([a, b]))
            zmf_part = self.evalmf(x, "zmf", torch.stack([c, d]))
            return torch.minimum(smf_part, zmf_part)

        else:
            raise NotImplementedError(f"MF type '{mf_type}' not implemented.")

    # ======================================================================
    # evalfis() — Mamdani 推理 + defuzz
    # ======================================================================
    def eval(self, inputs, point_n=101):
        """
        Mamdani 推理主函数（对应 fuzzyR::evalfis 的 Mamdani 情况的简化版）：
            1. 输入 fuzzification
            2. 规则激活（AND/OR）
            3. 输出 MF 裁剪/缩放 (implication)
            4. 规则聚合 (aggregation)
            5. defuzz 得到 crisp 输出（这里返回 tensor，以保留梯度）
        """
        # 训练时可以把这行 print 注释掉
        # print("=== Starting FIS Evaluation ===")

        # 把输入转成 1D tensor
        x = torch.as_tensor(inputs, dtype=torch.float32).flatten()

        device = x.device

        # ---- Step 1: Fuzzification ----
        input_mfs = []   # input_mfs[i][k] = 第 i 个 input 在第 k 个 MF 的 μ
        for i, var in enumerate(self.input):
            var_mf_vals = []
            for mf in var["mf"]:
                mu = self.evalmf(x[i], mf["type"], mf["params"]).to(device)
                var_mf_vals.append(mu)
            input_mfs.append(var_mf_vals)

        # ---- Step 2: Rule evaluation ----
        rule_strengths = []
        for rule in self.rule:
            antecedents = []
            for j in range(len(self.input)):
                val = rule[j]
                if val == 0:
                    continue
                mf_idx = abs(val) - 1
                mu = input_mfs[j][mf_idx]
                if val < 0:
                    mu = 1 - mu
                antecedents.append(mu)

            if not antecedents:
                firing = torch.zeros((), dtype=torch.float32, device=device)
            else:
                and_or = rule[-1]
                stack_ = torch.stack(antecedents)
                if and_or == 1:    # AND
                    firing = torch.min(stack_)
                else:              # OR
                    firing = torch.max(stack_)

            # 权重也保持为 tensor
            weight = torch.tensor(rule[-2], dtype=torch.float32, device=device)
            firing = firing * weight
            rule_strengths.append(firing)

        # ---- Step 3: Aggregation over output universe ----
        y = torch.linspace(self.output[0]["range"][0],
                           self.output[0]["range"][1],
                           point_n, dtype=torch.float32,
                           device=device)

        agg_mu = torch.zeros_like(y)

        for rule, firing in zip(self.rule, rule_strengths):
            # rule 的输出 MF index 在位置 len(self.input)
            out_idx = abs(rule[len(self.input)]) - 1
            out_mf = self.output[0]["mf"][out_idx]
            mu = self.evalmf(y, out_mf["type"], out_mf["params"]).to(device)

            # implication
            if self.imp_method == "min":
                mu = torch.minimum(mu, firing.expand_as(mu))
            elif self.imp_method == "prod":
                mu = mu * firing
            else:
                raise NotImplementedError(
                    f"Implication method '{self.imp_method}' not implemented."
                )

            # aggregation
            if self.agg_method == "max":
                agg_mu = torch.maximum(agg_mu, mu)
            elif self.agg_method == "sum":
                agg_mu = torch.clamp(agg_mu + mu, 0, 1)
            else:
                raise NotImplementedError(
                    f"Aggregation method '{self.agg_method}' not implemented."
                )

        # ---- Step 4: Defuzzification ----
        # 返回 tensor（0-dim），保持梯度
        crisp_output = self.defuzz(y, agg_mu, self.defuzz_method)
        # print("=== Evaluation Completed ===")
        return crisp_output

    # ======================================================================
    # defuzz() — defuzzification methods (全部返回 tensor)
    # ======================================================================
    def defuzz(self, x, mf, method="centroid"):
        """
        x: support (torch tensor, 1D)
        mf: membership values (same length as x)
        返回：0-dim tensor（scalar tensor），不做 .item()
        """
        # 不再 detach，保持梯度
        # x, mf = x, mf

        if method == "centroid":
            s = mf.sum()
            return (mf * x).sum() / s if s > 0 else x.mean()

        elif method == "bisector":
            cs = mf.cumsum(dim=0)
            total = mf.sum()
            if total <= 0:
                return x.mean()
            half = total / 2
            idx = (cs > half).nonzero(as_tuple=True)[0][0]
            return x[0] if idx == 0 else (x[idx - 1] + x[idx]) / 2.0

        elif method == "mom":
            maxv = mf.max()
            idx = (mf == maxv).nonzero(as_tuple=True)[0]
            return x[idx].mean()

        elif method == "som":
            maxv = mf.max()
            idx = (mf == maxv).nonzero(as_tuple=True)[0]
            return x[idx[0]]

        elif method == "lom":
            maxv = mf.max()
            idx = (mf == maxv).nonzero(as_tuple=True)[0]
            return x[idx[-1]]

        else:
            raise NotImplementedError(f"Defuzzification method '{method}' not implemented.")

    # ======================================================================
    # summary() — equivalent to fuzzyR::summary()
    # ======================================================================
    def summary(self):
        print("=== Fuzzy Inference System Summary ===")
        print(f"Name: {self.name}")
        print(f"Type: {self.type}")
        print(f"Inputs: {len(self.input)}")
        print(f"Outputs: {len(self.output)}")
        print(f"Rules: {len(self.rule)}")
        print(f"Defuzz Method: {self.defuzz_method}")
        print("======================================")

    # ======================================================================
    # plotmf() — 绘制某个变量所有 MF（只用于可视化，允许 numpy）
    # ======================================================================
    def plotmf(self, var_type="input", var_index=0, point_n=201, main=None):
        if var_type not in ["input", "output"]:
            raise ValueError("var_type must be 'input' or 'output'")

        var_list = self.input if var_type == "input" else self.output
        if var_index >= len(var_list):
            raise IndexError(f"{var_type}[{var_index}] does not exist")

        var = var_list[var_index]
        var_range = var["range"]
        x = torch.linspace(var_range[0], var_range[1], point_n).detach().cpu().numpy()

        plt.figure(figsize=(5, 3))

        for mf in var["mf"]:
            params = np.array(mf["params"])
            if mf["type"] == "gaussmf":
                sigma, c = params
                mu = np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
            else:
                raise NotImplementedError(f"MF type '{mf['type']}' not supported in plotmf().")

            plt.plot(x, mu, label=mf["name"], linewidth=2)

        plt.title(main or f"{var_type.capitalize()} {var['name']} MFs")
        plt.xlabel(var["name"])
        plt.ylabel("Membership degree")
        plt.grid(True)
        plt.legend()
        plt.show()

    # ======================================================================
    # plotvar() — 单输入下的输入→输出映射（可视化，不参与训练）
    # ======================================================================
    def plotvar(self, input_index=0, point_n=101):
        if len(self.input) == 0 or len(self.output) == 0:
            raise RuntimeError("FIS must have at least one input and one output.")
        if len(self.input) > 1:
            print("Warning: plotvar() currently supports single-input systems only.")

        var = self.input[input_index]
        x = np.linspace(var["range"][0], var["range"][1], point_n)
        # 这里 eval 返回 tensor，取 .item() 只用于画图，不影响训练
        y = [self.eval([float(v)], point_n=point_n).detach().cpu().item() for v in x]

        plt.figure(figsize=(5, 3))
        plt.plot(x, y, linewidth=2)
        plt.title(f"Fuzzy Relationship: {var['name']} → {self.output[0]['name']}")
        plt.xlabel(var["name"])
        plt.ylabel(self.output[0]["name"])
        plt.grid(True)
        plt.show()

    # ======================================================================
    # plot_graph() — 绘制模糊系统结构图（纯可视化）
    # ======================================================================
    def plot_graph(self):
        G = nx.DiGraph()

        input_nodes = [f"Input: {v['name']}" for v in self.input]
        rule_nodes = [f"Rule {i+1}" for i in range(len(self.rule))]
        output_nodes = [f"Output: {v['name']}" for v in self.output]

        G.add_nodes_from(input_nodes + rule_nodes + output_nodes)

        for r_i, rule in enumerate(self.rule):
            # 输入 → 规则
            for j, val in enumerate(rule[:len(self.input)]):
                if val != 0:
                    G.add_edge(input_nodes[j], rule_nodes[r_i])

            # 规则 → 输出
            out_pos = len(self.input)
            if out_pos < len(rule):
                out_idx = abs(rule[out_pos]) - 1
                if out_idx < len(output_nodes):
                    G.add_edge(rule_nodes[r_i], output_nodes[out_idx])

        # 简单分层布局
        pos = {}
        for i, node in enumerate(input_nodes):
            pos[node] = (0, -i)
        for i, node in enumerate(rule_nodes):
            pos[node] = (1, -i)
        for i, node in enumerate(output_nodes):
            pos[node] = (2, -i)

        plt.figure(figsize=(8, 4))
        nx.draw(G, pos, with_labels=True, arrows=True,
                node_color="#add8e6", node_size=1800,
                font_size=10, font_weight="bold")
        plt.title(f"Structure of FIS: {self.name}")
        plt.show()
