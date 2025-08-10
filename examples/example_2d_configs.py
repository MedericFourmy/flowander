from flowander.conditional_probability_paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearConditionalProbabilityPath, LinearMultisampleConditionalProbabilityPath, SquareRootBeta
from flowander.distributions import CheckerboardSampleable, CirclesSampleable, Gaussian, GaussianMixture, MoonsSampleable
from flowander.mlp import MLPVectorField


def config(model_name, device):
    # mostly used
    p_simple = Gaussian.isotropic(dim=2, std=1.0)
    flow_model = MLPVectorField(dim=2, hiddens=[64,64,64,64])
    alpha = LinearAlpha()
    beta = SquareRootBeta()

    match model_name:
        case "vf_gcpp_gaussian2GM":
            p_data = GaussianMixture.symmetric_2D(nmodes=5, std=1.0, scale=10.0).to(device)
            path = GaussianConditionalProbabilityPath(p_data, alpha, beta).to(device)

        case "vf_gcpp_gaussian2Moons":
            p_data = MoonsSampleable(device, scale=3.5)
            path = GaussianConditionalProbabilityPath(p_data, alpha, beta).to(device)

        case "vf_gcpp_gaussian2Circles":
            p_data = CirclesSampleable(device)
            path = GaussianConditionalProbabilityPath(p_data, alpha, beta).to(device)

        case "vf_gcpp_gaussian2Check":
            p_data = CheckerboardSampleable(device, grid_size=4)
            path = GaussianConditionalProbabilityPath(p_data, alpha, beta).to(device)

        case "vf_linear_mlp_gaussian2Checker":
            p_data = CheckerboardSampleable(device, grid_size=4)
            path = LinearConditionalProbabilityPath(p_simple, p_data).to(device)

        case "vf_linear_multi_mlp_gaussian2Checker":
            p_data = CheckerboardSampleable(device, grid_size=4)
            path = LinearMultisampleConditionalProbabilityPath(p_simple, p_data).to(device)

        case "vf_linear_mlp_circles2Checker":
            p_simple = CirclesSampleable(device)
            p_data = CheckerboardSampleable(device, grid_size=4)
            path = LinearConditionalProbabilityPath(p_simple, p_data).to(device)

        case "vf_linear_multi_mlp_circles2Checker":
            p_simple = CirclesSampleable(device)
            p_data = CheckerboardSampleable(device, grid_size=4)
            path = LinearMultisampleConditionalProbabilityPath(p_simple, p_data).to(device)

        case _:
            raise ValueError(f"{model_name} is not a valid example config")


    return p_simple, p_data, path, flow_model

