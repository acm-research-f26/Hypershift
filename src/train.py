import torch

def train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total_loss = 0.0
    target_count = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        edge_weight = getattr(batch, "edge_weight", None)
        predictions = model(
            batch.x,
            batch.edge_index,
            edge_weight=edge_weight,
        )
        loss = loss_fn(predictions, batch.y)
        loss.backward()
        optimizer.step()
        batch_target_count = batch.y.numel()
        total_loss += float(loss.detach().item()) * batch_target_count
        target_count += batch_target_count

    if target_count == 0:
        raise ValueError("Training loader produced no targets.")
    
    return total_loss / target_count

def train(
    model,
    train_loader,
    validation_loader,
    loss_fn,
    *,
    epochs: int = 100,
    patience: int = 15,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    tolerance: float | None = None,
    device=None,
):
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")
    if tolerance is not None and tolerance <= 0:
        raise ValueError("tolerance must be positive when provided.")
    resolved_device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_losses: list[float] = []
    validation_mses: list[float] = []   
    validation_maes: list[float] = []
    validation_directional_accuracies: list[float] = []
    parameter_change_inf_norms: list[float] = []
    best_validation_mse = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_ran = 0
    stopped_by_tolerance = False
    for epoch in range(1, epochs + 1):
        parameters_before_epoch: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameters_before_epoch[name] = parameter.detach().clone()

        training_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            resolved_device,
        )

        # $\lVert\theta_e-\theta_{e-1}\rVert_\infty = \max_j\lvert\theta_{e,j}-\theta_{e-1,j}\rvert$
        parameter_change_inf_norm = 0.0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            parameter_change = (parameter.detach() - parameters_before_epoch[name])
            parameter_tensor_inf_norm = float(torch.linalg.vector_norm(parameter_change, ord=float("inf"),).item())
            if parameter_tensor_inf_norm > parameter_change_inf_norm:
                parameter_change_inf_norm = parameter_tensor_inf_norm

        validation_metrics = evaluate(
            model,
            validation_loader,
            resolved_device,
        )

        train_losses.append(training_loss)
        validation_mses.append(validation_metrics["mse"])
        validation_maes.append(validation_metrics["mae"])
        validation_directional_accuracies.append(
            validation_metrics["directional_accuracy"]
        )
        parameter_change_inf_norms.append(parameter_change_inf_norm)

        epochs_ran = epoch
        validation_mse = validation_metrics["mse"]

        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            epochs_without_improvement = 0

            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if tolerance is not None and parameter_change_inf_norm < tolerance:
            stopped_by_tolerance = True
            break

        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    return {
        "train_loss": train_losses,
        "validation_mse": validation_mses,
        "validation_mae": validation_maes,
        "validation_directional_accuracy": validation_directional_accuracies,
        "parameter_change_inf_norm": parameter_change_inf_norms,
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_mse,
        "epochs_ran": epochs_ran,
        "stopped_by_tolerance": stopped_by_tolerance,
    }

@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    total_squared_error  = 0.0
    total_absolute_error = 0.0
    correct_directions = 0
    target_count = 0

    for batch in loader:
        batch = batch.to(device)
        edge_weight = getattr(batch, "edge_weight", None)
        preds = model(batch.x, batch.edge_index, edge_weight=edge_weight)
        errors = preds - batch.y
        total_squared_error += errors.square().sum().item()
        total_absolute_error += errors.abs().sum().item()

        predicted_direction = preds >= 0
        target_direction = batch.y >= 0
        correct_directions += (predicted_direction == target_direction).sum().item()
        target_count += batch.y.numel()

    if target_count == 0:
        raise ValueError("Eval loader produced no targets")
    return {
        "mse": total_squared_error / target_count,
        "mae": total_absolute_error / target_count,
        "directional_accuracy": correct_directions / target_count,
    }
        
