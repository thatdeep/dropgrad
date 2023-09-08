import torch
import torch.nn as nn
import torch.nn.functional as F



def train_loop(args, model, train_loader, optimizer, epoch):
    device = args.device
    model.train()
    
    if hasattr(args, 'dropgrad'):
        # Dictionary to store dropped gradients' indices
        dropped_tower = {}
        # tower dimension
        max_time_steps = args.dropgrad.max_time_steps
        drop_momentum = args.dropgrad.decay

        for name, param in model.named_parameters():
            dropped_tower[name] = torch.zeros((max_time_steps,) + param.shape, device=device)
    
    total_batches_per_epoch = len(train_loader)

    iteration = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        if hasattr(args, 'dropgrad'):
            # Initialize dropout layer
            p = args.dropgrad.bernoulli_prob
            current_prob = next((item['prob'] for item in reversed(args.dropgrad.dropout_schedule) if item['epoch'] <= epoch), p)
            
            dropout = nn.Dropout(p=current_prob)

            # Apply dropout and store dropped gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Apply dropout and get mask
                    mask = (dropout(torch.ones_like(param.grad)) > 0).float()
                    if hasattr(args.dropgrad, 'forget') and args.dropgrad.forget:
                        param.grad *= mask
                        continue
                    dropped = param.grad * (1.0 - mask)
                    
                    # Create an empty tensor B with shape (b, *A_shape)
                    dropped_shape = (max_time_steps,) + mask.shape
                    dropped_placeholder = torch.zeros(dropped_shape, device=device)

                    # Generate random k-indices for each element in A
                    k_indices = torch.randint(1, max_time_steps, mask.shape)

                    # Create index arrays for all dimensions
                    index_arrays = [torch.arange(size) for size in mask.shape]
                    meshgrid_indices = torch.meshgrid(*index_arrays)

                    # Add k_indices to the list of meshgrid indices
                    meshgrid_indices = (k_indices, *meshgrid_indices)

                    # Use advanced indexing to fill B
                    dropped_placeholder[meshgrid_indices] = dropped

                    # Apply mask to gradients
                    param.grad *= mask
                    
                    # what we do now is unload grad tower by 1 level, and load it with new masked grads

                    if hasattr(args.dropgrad, 'partial') and args.dropgrad.partial:
                        param.grad += drop_momentum * dropped_tower[name][0, ...] * mask
                        dropped_tower[name][1, ...] += dropped_tower[name][0, ...] * (1.0 - mask)
                    else:
                        param.grad += drop_momentum * dropped_tower[name][0, ...]
                    
                    dropped_tower[name][0:max_time_steps - 1, ...] = drop_momentum * dropped_tower[name][1: max_time_steps, ...]
                    dropped_tower[name][max_time_steps - 1, ...] = 0
                    
                    dropped_tower[name] += dropped_placeholder
                
        optimizer.step()
        iteration += 1
        if batch_idx % args.logging.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # Log to TensorBoard
            args.writer.add_scalar('Train/Loss', loss.item(), (epoch - 1) * total_batches_per_epoch + iteration)
            if args.debug.dry_run:
                break


def test(args, model, test_loader, epoch):
    device = args.device
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = torch.max(output, 1).indices
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    args.writer.add_scalar('Test/Accuracy', 100. * correct / len(test_loader.dataset), epoch)
    args.writer.add_scalar('Test/Loss', test_loss, epoch)
