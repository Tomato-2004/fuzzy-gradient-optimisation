import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch.nn.functional as F 


# ============================================================
# Utility: Safe reparameterization for triangle/trapezoid MFs
# ============================================================
def reparam_trimf(params):
    p = torch.as_tensor(params, dtype=torch.float32)
    a = p[0]
    b = a + torch.nn.functional.softplus(p[1])
    c = b + torch.nn.functional.softplus(p[2])
    return torch.stack([a, b, c])


def reparam_trapmf(params):
    p = torch.as_tensor(params, dtype=torch.float32)
    a = p[0]
    b = a + torch.nn.functional.softplus(p[1])
    c = b + torch.nn.functional.softplus(p[2])
    d = c + torch.nn.functional.softplus(p[3])
    return torch.stack([a, b, c, d])


# ============================================================
# FuzzyInferenceSystem (Mamdani Type-1, tensorized)
# ============================================================
class FuzzyInferenceSystem:
    """
    完全保留原有功能（fuzzyR接口、绘图、eval单样本）。
    新增 eval_batch() 用于训练时提升速度。
    """

    def __init__(self, name, fis_type="mamdani", mf_type="t1",
                 and_method="min", or_method="max",
                 imp_method="min", agg_method="max",
                 defuzz_method="centroid",
                 device=None):

        self.name = name
        self.type = fis_type
        self.mf_type = mf_type

        self.and_method = and_method
        self.or_method = or_method
        self.imp_method = imp_method
        self.agg_method = agg_method
        self.defuzz_method = defuzz_method

        self.input = []
        self.output = []
        self.rule = []

        self.device = device if device is not None else "cpu"

        print(f"Created FIS: {self.name} (type={self.type})")

    # ============================================================
    # fuzzyR::addvar()
    # ============================================================
    def add_variable(self, var_type, name, var_range,
                     method=None, params=None, firing_method=None):

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
            raise ValueError("var_type must be input or output")

        print(f"Added {var_type}: {name} range={var_range}")

    # ============================================================
    # fuzzyR::addmf()
    # ============================================================
    def add_mf(self, var_type, var_index, mf_name, mf_type, mf_params):

        mf_data = {
            "name": mf_name,
            "type": mf_type,
            "params": torch.tensor(mf_params, dtype=torch.float32)
        }

        if var_type == "input":
            self.input[var_index]["mf"].append(mf_data)
        else:
            self.output[var_index]["mf"].append(mf_data)

        print(f"Added MF '{mf_name}' ({mf_type}) to {var_type}[{var_index}]")

    # ============================================================
    # fuzzyR::addrule()
    # ============================================================
    def add_rule(self, rule_list):
        self.rule.append(rule_list)
        print(f"Added rule: {rule_list}")

    # ============================================================
    # t-norm / s-norm
    # ============================================================
    def tnorm(self, a, b):
        if self.and_method == "min":
            return torch.min(a, b)
        elif self.and_method == "prod":
            return a * b
        elif self.and_method == "lukasiewicz":
            return torch.clamp(a + b - 1, 0, 1)
        else:
            raise NotImplementedError

    def snorm(self, a, b):
        if self.or_method == "max":
            return torch.max(a, b)
        elif self.or_method == "prob_or":
            return a + b - a * b
        elif self.or_method == "lukasiewicz":
            return torch.clamp(a + b, 0, 1)
        else:
            raise NotImplementedError

    # ============================================================
    # evalmf (safe param)
    # ============================================================
    def evalmf(self, x, mf_type, params):

        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        if mf_type.startswith("gaussmf_casp"):
    # params = [raw_sigma, raw_center]
            raw_sigma = params[0]
            raw_center = params[1]

    # CASP reparameterization
            sigma = F.softplus(raw_sigma) + 1e-4
            center = raw_center

    # 正确的 Gaussian MF 计算
            return torch.exp(-((x - center) ** 2) / (2 * sigma ** 2))


        if mf_type == "trimf":
            p = reparam_trimf(params)
        elif mf_type == "trapmf":
            p = reparam_trapmf(params)
        else:
            p = torch.as_tensor(params, dtype=torch.float32, device=self.device)

        if mf_type == "gaussmf":
            sigma, c = p
            return torch.exp(-((x - c)**2) / (2 * sigma**2))

        elif mf_type == "gbellmf":
            a, b, c = p
            return 1 / (1 + (torch.abs((x - c) / a))**(2 * b))

        elif mf_type == "trimf":
            a, b, c = p
            left = (x - a) / (b - a + 1e-12)
            right = (c - x) / (c - b + 1e-12)
            return torch.clamp(torch.min(left, right), 0, 1)

        elif mf_type == "trapmf":
            a, b, c, d = p
            y1 = (x - a) / (b - a + 1e-12)
            y2 = torch.ones_like(x)
            y3 = (d - x) / (d - c + 1e-12)
            return torch.clamp(torch.min(torch.min(y1, y2), y3), 0, 1)

        elif mf_type == "sigmf":
            a, c = p
            return 1/(1 + torch.exp(-a*(x-c)))

        elif mf_type == "smf":
            a, b = p
            y = torch.zeros_like(x)
            idx2 = (x > a) & (x < (a+b)/2)
            idx3 = (x >= (a+b)/2) & (x < b)
            idx4 = x >= b
            y[idx2] = 2*((x[idx2]-a)/(b-a))**2
            y[idx3] = 1 - 2*((x[idx3]-b)/(b-a))**2
            y[idx4] = 1
            return y

        elif mf_type == "zmf":
            a, b = p
            y = torch.zeros_like(x)
            idx1 = x <= a
            idx2 = (x > a) & (x < (a+b)/2)
            idx3 = (x >= (a+b)/2) & (x < b)
            y[idx1] = 1
            y[idx2] = 1 - 2*((x[idx2]-a)/(b-a))**2
            y[idx3] = 2*((x[idx3]-b)/(b-a))**2
            return y

        elif mf_type == "pimf":
            a, b, c, d = p
            sm = self.evalmf(x, "smf", torch.tensor([a, b]))
            zm = self.evalmf(x, "zmf", torch.tensor([c, d]))
            return torch.min(sm, zm)

        else:
            raise NotImplementedError

    # ============================================================
    # evalfis (single-sample, fuzzyR-compatible)
    # ============================================================
    def eval(self, inputs, point_n=101):

        x = torch.as_tensor(inputs, dtype=torch.float32, device=self.device).flatten()
        device = self.device

        # fuzzification
        input_mfs = []
        for i, var in enumerate(self.input):
            row = []
            for mf in var["mf"]:
                mu = self.evalmf(x[i], mf["type"], mf["params"])
                row.append(mu)
            input_mfs.append(row)

        # rule evaluation
        rule_strengths = []
        for rule in self.rule:
            antecedents = []
            for j in range(len(self.input)):
                idx = rule[j]
                if idx == 0:
                    continue
                mf_idx = abs(idx)-1
                mu = input_mfs[j][mf_idx]
                if idx < 0:
                    mu = 1-mu
                antecedents.append(mu)

            if len(antecedents)==0:
                firing = torch.tensor(0.0, device=device)
            else:
                if rule[-1]==1:
                    firing = torch.stack(antecedents).min()
                else:
                    firing = torch.stack(antecedents).max()

            firing = firing * torch.tensor(rule[-2], device=device)
            rule_strengths.append(firing)

        # aggregation universe
        y = torch.linspace(self.output[0]["range"][0],
                           self.output[0]["range"][1],
                           point_n, device=device)

        agg_mu = torch.zeros_like(y)

        for rule, firing in zip(self.rule, rule_strengths):
            out_idx = abs(rule[len(self.input)])-1
            mf = self.output[0]["mf"][out_idx]
            mu = self.evalmf(y, mf["type"], mf["params"])

            if self.imp_method=="min":
                mu = torch.min(mu, firing)
            else:
                mu = mu * firing

            if self.agg_method=="max":
                agg_mu = torch.max(agg_mu, mu)
            else:
                agg_mu = torch.clamp(agg_mu + mu, 0, 1)

        return self.defuzz(y, agg_mu, self.defuzz_method)

    # ============================================================
    # defuzz
    # ============================================================
    def defuzz(self, x, mf, method="centroid"):

        if method=="centroid":
            s = mf.sum()
            return (mf*x).sum()/(s+1e-12)

        elif method=="bisector":
            cs = mf.cumsum(0)
            total = cs[-1]
            half = total/2
            idx = (cs>half).nonzero(as_tuple=True)[0][0]
            if idx==0:
                return x[0]
            return (x[idx-1]+x[idx])/2

        elif method=="mom":
            maxv = mf.max()
            idx = (mf==maxv).nonzero(as_tuple=True)[0]
            return x[idx].mean()

        elif method=="som":
            maxv = mf.max()
            idx = (mf==maxv).nonzero(as_tuple=True)[0]
            return x[idx[0]]

        elif method=="lom":
            maxv = mf.max()
            idx = (mf==maxv).nonzero(as_tuple=True)[0]
            return x[idx[-1]]

        else:
            raise NotImplementedError

    # ============================================================
    # summary
    # ============================================================
    def summary(self):
        print("=== FIS SUMMARY ===")
        print("Name:", self.name)
        print("#Inputs:", len(self.input))
        print("#Outputs:", len(self.output))
        print("#Rules:", len(self.rule))
        print("Defuzz:", self.defuzz_method)
        print("====================")

    # ============================================================
    # plotmf
    # ============================================================
    def plotmf(self, var_type="input", var_index=0, point_n=201, main=None):

        if var_type=="input":
            var = self.input[var_index]
        else:
            var = self.output[var_index]

        r0, r1 = var["range"]
        x = np.linspace(r0, r1, point_n)

        plt.figure(figsize=(5,3))
        for mf in var["mf"]:
            MF = self.evalmf(torch.tensor(x), mf["type"], mf["params"]).detach().cpu().numpy()
            plt.plot(x, MF, label=mf["name"], linewidth=2)
        plt.title(main or f"{var_type.capitalize()} {var['name']} MFs")
        plt.legend()
        plt.grid()
        plt.show()

    # ============================================================
    # plotvar
    # ============================================================
    def plotvar(self, input_index=0, point_n=101):

        if len(self.input) > 1:
            print("Warning: plotvar supports one input only.")

        var = self.input[input_index]
        x = np.linspace(var["range"][0], var["range"][1], point_n)
        y = [self.eval([float(v)]).item() for v in x]

        plt.figure(figsize=(5,3))
        plt.plot(x, y)
        plt.grid()
        plt.title(f"{var['name']} → {self.output[0]['name']}")
        plt.show()

    # ============================================================
    # plot_graph (修复版本)
    # ============================================================
    def plot_graph(self):
        G = nx.DiGraph()

        # 输入变量节点
        input_nodes = [f"Input: {v['name']}" for v in self.input]

        # 输出变量节点
        output_var_nodes = [f"OutputVar: {v['name']}" for v in self.output]

        # 输出 MF 节点
        output_mf_nodes = []
        for ov in self.output:
            for mf in ov["mf"]:
                output_mf_nodes.append(f"MF: {mf['name']}")

        # 规则节点
        rule_nodes = [f"Rule {i+1}" for i in range(len(self.rule))]

        G.add_nodes_from(input_nodes + rule_nodes + output_var_nodes + output_mf_nodes)

        # 输入 → 规则
        for r_i, rule in enumerate(self.rule):
            for j, val in enumerate(rule[:len(self.input)]):
                if val != 0:
                    G.add_edge(input_nodes[j], rule_nodes[r_i])

            # 规则 → 输出 MF
            cons_idx = abs(rule[len(self.input)]) - 1
            mf_name = self.output[0]["mf"][cons_idx]["name"]
            G.add_edge(rule_nodes[r_i], f"MF: {mf_name}")

        # 输出变量 → 输出 MF
        for ov in self.output:
            var_node = f"OutputVar: {ov['name']}"
            for mf in ov["mf"]:
                G.add_edge(var_node, f"MF: {mf['name']}")

        # 节点布局
        pos = {}

        # 输入在左
        for i, node in enumerate(input_nodes):
            pos[node] = (0, -i)

        # 规则在中
        for i, node in enumerate(rule_nodes):
            pos[node] = (1, -i)

        # 输出变量靠右
        for i, node in enumerate(output_var_nodes):
            pos[node] = (2, -(i*2))

        # 输出 MF 最右
        for i, node in enumerate(output_mf_nodes):
            pos[node] = (3, -i)

        plt.figure(figsize=(10,5))
        nx.draw(G, pos, with_labels=True, arrows=True,
                node_color="#ADD8E6", node_size=1700,
                font_size=9, font_weight="bold")
        plt.title(f"Structure of FIS: {self.name}")
        plt.show()

    # ============================================================
    # NEW: eval_batch (vectorized, FAST for training)
    # ============================================================
    def eval_batch(self, X, point_n=101):
        """
        X: (batch, n_inputs)
        return: (batch,) tensor
        NOTE: 完全 vectorized，不改变 eval() 行为，只用于训练
        """
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        batch, n_inputs = X.shape

        # ---------- fuzzification ----------
        # input_mfs[i][k] shape -> (batch,)
        input_mfs = []
        for i, var in enumerate(self.input):
            row = []
            x_i = X[:, i]
            for mf in var["mf"]:
                mu = self.evalmf(x_i, mf["type"], mf["params"])
                row.append(mu)
            input_mfs.append(row)

        # ---------- rule firing ----------
        fires = []
        for rule in self.rule:
            antecedents = []
            for j in range(n_inputs):
                idx = rule[j]
                if idx == 0:
                    continue
                mf_idx = abs(idx) - 1
                mu = input_mfs[j][mf_idx]
                if idx < 0:
                    mu = 1 - mu
                antecedents.append(mu)

            if len(antecedents) == 0:
                firing = torch.zeros(batch, device=self.device)
            else:
                stack = torch.stack(antecedents, dim=0)
                if rule[-1] == 1:
                    firing = stack.min(dim=0).values
                else:
                    firing = stack.max(dim=0).values

            firing = firing * rule[-2]
            fires.append(firing)

        fires = torch.stack(fires, dim=0)  # (n_rules, batch)

        # ---------- aggregation ----------
        y = torch.linspace(
            self.output[0]["range"][0],
            self.output[0]["range"][1],
            point_n,
            device=self.device,
        )
        P = point_n
        agg = torch.zeros(batch, P, device=self.device)

        for r_idx, rule in enumerate(self.rule):
            cons_idx = abs(rule[n_inputs]) - 1
            mf = self.output[0]["mf"][cons_idx]

            mu_y = self.evalmf(y, mf["type"], mf["params"])
            firing = fires[r_idx].unsqueeze(1)

            if self.imp_method == "min":
                applied = torch.min(mu_y.unsqueeze(0), firing)
            else:
                applied = mu_y.unsqueeze(0) * firing

            if self.agg_method == "max":
                agg = torch.max(agg, applied)
            else:
                agg = torch.clamp(agg + applied, 0, 1)

        # ---------- defuzz ----------
        num = (agg * y.unsqueeze(0)).sum(dim=1)
        den = agg.sum(dim=1) + 1e-12
        return num / den
