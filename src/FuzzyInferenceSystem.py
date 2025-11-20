import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class FuzzyInferenceSystem:
    """
    Python reimplementation of FuzzyR::fuzzyinferencesystem.R
    Logical order: newfis() → addvar() → addmf() → addrule() → evalfis() + defuzz()
    """

    def __init__(self, name, fis_type="mamdani", mf_type="t1",
                 and_method="min", or_method="max", imp_method="min",
                 agg_method="max", defuzz_method="centroid"):
        """Equivalent to fuzzyR::newfis()"""
        self.name = name
        self.type = fis_type          # "mamdani" / "ts" / "casp" 等
        self.mf_type = mf_type
        self.and_method = and_method
        self.or_method = or_method
        self.imp_method = imp_method
        self.agg_method = agg_method
        self.defuzz_method = defuzz_method

        self.input = []
        self.output = []
        self.rule = []
        print(f"Created FIS: {self.name} (type={self.type})")

    # --------------------------------------------------------------
    # addvar()
    # --------------------------------------------------------------
    def add_variable(self, var_type, name, var_range,
                     method=None, params=None, firing_method="tnorm.min.max"):
        variable = {
            "name": name,
            "range": list(var_range),
            "method": method,
            "params": params,
            "mf": [],
            "firing_method": firing_method
        }
        if var_type == "input":
            self.input.append(variable)
        elif var_type == "output":
            self.output.append(variable)
        else:
            raise ValueError("var_type must be 'input' or 'output'")
        print(f"Added {var_type}: {name} range={var_range}")

    # --------------------------------------------------------------
    # addmf()
    # --------------------------------------------------------------
    def add_mf(self, var_type, var_index, mf_name, mf_type, mf_params):
        mf_data = {"name": mf_name, "type": mf_type, "params": mf_params}
        if var_type == "input":
            self.input[var_index]["mf"].append(mf_data)
        elif var_type == "output":
            self.output[var_index]["mf"].append(mf_data)
        else:
            raise ValueError("var_type must be 'input' or 'output'")
        print(f"Added MF '{mf_name}' ({mf_type}) to {var_type}[{var_index}]")

    # --------------------------------------------------------------
    # addrule()
    # --------------------------------------------------------------
    def add_rule(self, rule_list):
        """
        Mamdani 模式：rule_list 是 list / matrix 行，
            如 [2, 2, 1, 1] 这种 fuzzyR 风格。
        TS / CASP 模式：你可以继续直接操作 self.rule，存 dict 形式的规则：
            {"antecedent": [...], "coeff": [...]}
        这个函数只简单 append，不强制格式。
        """
        self.rule.append(rule_list)
        print(f"Added rule: {rule_list}")

    # --------------------------------------------------------------
    # showrule()
    # --------------------------------------------------------------
    def show_rules(self):
        if not self.rule:
            print("No rules defined.")
            return
        print("=== Rule Base ===")
        for i, r in enumerate(self.rule, 1):
            print(f"{i}. Rule: {r}")
        print("=================")

    # --------------------------------------------------------------
    # evalmf()
    # --------------------------------------------------------------
    def evalmf(self, x, mf_type, params):
        x = torch.tensor(x, dtype=torch.float32)
        if mf_type == "gaussmf":
            sigma, c = params
            return torch.exp(-((x - c) ** 2) / (2 * sigma ** 2))
        elif mf_type == "gbellmf":
            a, b, c = params
            return 1 / (1 + ((torch.abs((x - c) / a)) ** (2 * b)))
        elif mf_type == "trimf":
            a, b, c = params
            return torch.clamp(torch.minimum((x - a) / (b - a),
                                             (c - x) / (c - b)), 0, 1)
        elif mf_type == "trapmf":
            a, b, c, d = params
            y1 = (x - a) / (b - a)
            y2 = torch.ones_like(x)
            y3 = (d - x) / (d - c)
            y = torch.minimum(torch.minimum(y1, y2), y3)
            return torch.clamp(y, 0, 1)
        elif mf_type == "sigmf":
            a, c = params
            return 1 / (1 + torch.exp(-a * (x - c)))
        elif mf_type == "smf":
            a, b = params
            y = torch.zeros_like(x)
            idx1 = x <= a
            idx2 = (x > a) & (x < (a + b) / 2)
            idx3 = (x >= (a + b) / 2) & (x < b)
            idx4 = x >= b
            y[idx2] = 2 * ((x[idx2] - a) / (b - a)) ** 2
            y[idx3] = 1 - 2 * ((x[idx3] - b) / (b - a)) ** 2
            y[idx4] = 1
            return y
        elif mf_type == "zmf":
            a, b = params
            y = torch.zeros_like(x)
            idx1 = x <= a
            idx2 = (x > a) & (x < (a + b) / 2)
            idx3 = (x >= (a + b) / 2) & (x < b)
            idx4 = x >= b
            y[idx1] = 1
            y[idx2] = 1 - 2 * ((x[idx2] - a) / (b - a)) ** 2
            y[idx3] = 2 * ((x[idx3] - b) / (b - a)) ** 2
            return y
        elif mf_type == "pimf":
            a, b, c, d = params
            smf_part = self.evalmf(x, "smf", [a, b])
            zmf_part = self.evalmf(x, "zmf", [c, d])
            return torch.minimum(smf_part, zmf_part)
        else:
            raise NotImplementedError(f"MF type '{mf_type}' not implemented.")

    # --------------------------------------------------------------
    # evalfis() 入口：Mamdani / TS 自动切换
    # --------------------------------------------------------------
    def eval(self, inputs, point_n=101):
        # TS / CASP 走可微 TS 分支（与你 notebook 测试一致）
        if self.type is not None and self.type.lower() in ["ts", "casp"]:
            return self.eval_ts(inputs)

        # ======= 下面是原来的 Mamdani 逻辑，保持不变 =======
        #print("=== Starting FIS Evaluation ===")
        x = torch.tensor(inputs, dtype=torch.float32).flatten()

        # ---- Step 1: Fuzzification ----
        input_mfs = []
        for i, var in enumerate(self.input):
            var_mf_vals = []
            for mf in var["mf"]:
                mu = self.evalmf(x[i], mf["type"], mf["params"])
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
            # rule[-1]：1=min，0=max（与原实现保持一致）
            firing = torch.min(torch.stack(antecedents)) if rule[-1] == 1 \
                     else torch.max(torch.stack(antecedents))
            firing *= float(rule[-2])  # rule[-2]：权重
            rule_strengths.append(firing)

        # ---- Step 3: Aggregation ----
        y = torch.linspace(self.output[0]["range"][0],
                           self.output[0]["range"][1],
                           point_n, dtype=torch.float32)
        agg_mu = torch.zeros_like(y)

        for rule, firing in zip(self.rule, rule_strengths):
            out_idx = abs(rule[len(self.input)]) - 1
            out_mf = self.output[0]["mf"][out_idx]
            mu = self.evalmf(y, out_mf["type"], out_mf["params"])

            # Implication
            if self.imp_method == "min":
                mu = torch.minimum(mu, torch.tensor(firing))
            elif self.imp_method == "prod":
                mu = mu * firing

            # Aggregation
            if self.agg_method == "max":
                agg_mu = torch.maximum(agg_mu, mu)
            elif self.agg_method == "sum":
                agg_mu = torch.clamp(agg_mu + mu, 0, 1)

        # ---- Step 4: Defuzzification ----
        crisp_output = self.defuzz(y, agg_mu, self.defuzz_method)
        #print("=== Evaluation Completed ===")
        return crisp_output

    # --------------------------------------------------------------
    # defuzz()
    # --------------------------------------------------------------
    def defuzz(self, x, mf, method="centroid"):
        x, mf = x.detach(), mf.detach()
        if method == "centroid":
            s = torch.sum(mf)
            return (torch.sum(mf * x) / s).item() if s > 0 else torch.mean(x).item()
        elif method == "bisector":
            cs, total = torch.cumsum(mf, dim=0), torch.sum(mf)
            if total == 0:
                return torch.mean(x).item()
            half = total / 2
            idx = torch.nonzero(cs > half, as_tuple=True)[0][0]
            return x[0].item() if idx == 0 else ((x[idx - 1] + x[idx]) / 2.0).item()
        elif method == "mom":
            maxv, idx = torch.max(mf), torch.nonzero(mf == torch.max(mf), as_tuple=True)[0]
            return torch.mean(x[idx]).item()
        elif method == "som":
            idx = torch.nonzero(mf == torch.max(mf), as_tuple=True)[0]
            return x[idx[0]].item()
        elif method == "lom":
            idx = torch.nonzero(mf == torch.max(mf), as_tuple=True)[0]
            return x[idx[-1]].item()
        else:
            raise NotImplementedError(f"Defuzzification method '{method}' not implemented.")

    # --------------------------------------------------------------
    # summary()
    # --------------------------------------------------------------
    def summary(self):
        print("=== Fuzzy Inference System Summary ===")
        print(f"Name: {self.name}")
        print(f"Type: {self.type}")
        print(f"Inputs: {len(self.input)}")
        print(f"Outputs: {len(self.output)}")
        print(f"Rules: {len(self.rule)}")
        print(f"Defuzz Method: {self.defuzz_method}")
        print("======================================")

    # --------------------------------------------------------------
    # plotmf()
    # --------------------------------------------------------------
    def plotmf(self, var_type="input", var_index=0, point_n=201, main=None):
        if var_type not in ["input", "output"]:
            raise ValueError("var_type must be 'input' or 'output'")

        var_list = self.input if var_type == "input" else self.output
        if var_index >= len(var_list):
            raise IndexError(f"{var_type}[{var_index}] does not exist")

        var = var_list[var_index]
        var_range = var["range"]
        x = torch.linspace(var_range[0], var_range[1], point_n).numpy()

        plt.figure(figsize=(5, 3))
        for mf in var["mf"]:
            params = np.array(mf["params"])
            if mf["type"] == "gaussmf":
                sigma, c = params
                mu = np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
            else:
                raise NotImplementedError(f"MF type '{mf['type']}' not supported in plotmf()")
            plt.plot(x, mu, label=mf["name"], linewidth=2)

        plt.title(main or f"{var_type.capitalize()} {var['name']} MFs")
        plt.xlabel(var["name"])
        plt.ylabel("Membership degree")
        plt.legend()
        plt.grid(True)
        plt.show()

    # --------------------------------------------------------------
    # plotvar()
    # --------------------------------------------------------------
    def plotvar(self, input_index=0, point_n=101):
        if len(self.input) == 0 or len(self.output) == 0:
            raise RuntimeError("FIS must have at least one input and one output.")
        if len(self.input) > 1:
            print("Warning: plotvar() currently supports single-input systems only.")

        var = self.input[input_index]
        x = np.linspace(var["range"][0], var["range"][1], point_n)
        y = [self.eval([v], point_n=point_n) for v in x]

        plt.figure(figsize=(5, 3))
        plt.plot(x, y, color="blue", linewidth=2)
        plt.title(f"Fuzzy Relationship: {var['name']} → {self.output[0]['name']}")
        plt.xlabel(var["name"])
        plt.ylabel(self.output[0]["name"])
        plt.grid(True)
        plt.show()

    # --------------------------------------------------------------
    # plot_graph()
    # --------------------------------------------------------------
    def plot_graph(self):
        G = nx.DiGraph()
        input_nodes = [f"Input: {v['name']}" for v in self.input]
        rule_nodes = [f"Rule {i+1}" for i in range(len(self.rule))]
        output_nodes = [f"Output: {v['name']}" for v in self.output]
        G.add_nodes_from(input_nodes + rule_nodes + output_nodes)

        for r_i, rule in enumerate(self.rule):
            for j, val in enumerate(rule[:len(self.input)]):
                if val != 0:
                    G.add_edge(input_nodes[j], rule_nodes[r_i])

            out_pos = len(self.input)
            if out_pos < len(rule):
                out_idx = abs(rule[out_pos]) - 1
                if out_idx < len(output_nodes):
                    G.add_edge(rule_nodes[r_i], output_nodes[out_idx])

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

    # ============================================================
    #  新增：TS / CASP（可微）推理引擎
    # ============================================================
    def eval_ts(self, inputs):
        """
        Takagi-Sugeno / CASP-like forward evaluation.

        规则格式（与你 notebook 里一致）：
        self.rule = [
            {
              "antecedent": [mf_idx_for_x1, mf_idx_for_x2, ...],  # 每个输入选择一个 MF
              "coeff":     [b0, b1, ..., bn] or torch tensor       # 线性后件 y = b0 + Σ bi xi
            },
            ...
        ]
        """
        # ---- 1. 输入：支持 list / numpy / tensor ----
        if isinstance(inputs, torch.Tensor):
            x = inputs
        else:
            x = torch.tensor(inputs, dtype=torch.float32)

        if x.dim() == 1:      # 单样本 → (1, num_features)
            x = x.unsqueeze(0)

        batch = x.shape[0]

        # ---- 2. 每个输入在每个 MF 上的 μ(x) ----
        mus = []
        for i, var in enumerate(self.input):
            mf_vals = []
            for mf in var["mf"]:
                mu = self.evalmf(x[:, i], mf["type"], mf["params"])
                mf_vals.append(mu)
            mus.append(mf_vals)

        # ---- 3. 规则激活强度（乘积 t-norm）----
        W_list = []
        for rule in self.rule:
            w = torch.ones(batch, dtype=torch.float32)
            for j, mf_idx in enumerate(rule["antecedent"]):
                w = w * mus[j][mf_idx]
            W_list.append(w)

        W = torch.stack(W_list, dim=1)  # (batch, num_rules)
        W_sum = W.sum(dim=1, keepdim=True) + 1e-8

        # ---- 4. 线性后件 y_r = b0 + Σ bi * xi ----
        Y_rules = []
        for rule in self.rule:
            coeff = rule["coeff"]
            if isinstance(coeff, torch.Tensor):
                coef = coeff
            else:
                coef = torch.tensor(coeff, dtype=torch.float32)

            # coef[0] = b0, coef[1:] = bi
            y_r = coef[0] + torch.sum(coef[1:] * x, dim=1)
            Y_rules.append(y_r)

        Y_rules = torch.stack(Y_rules, dim=1)  # (batch, num_rules)

        # ---- 5. 加权平均 ----
        y = torch.sum(W * Y_rules, dim=1) / W_sum.squeeze(1)

        # 返回 tensor，保留梯度；单样本时是 0-dim tensor
        return y if batch > 1 else y.squeeze(0)
