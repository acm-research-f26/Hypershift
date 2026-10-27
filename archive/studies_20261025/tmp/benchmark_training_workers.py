from concurrent.futures import ProcessPoolExecutor
import time

def measure(seed):
    import torch
    from training_device import configure_device
    from robust_models import make_model,robust_objective
    device=configure_device('cuda')
    torch.manual_seed(seed)
    model=make_model('think_mix_huber',101,[list(range(101))]).to(device)
    x=torch.randn(8,16,101,5,device=device)
    mask=torch.ones(8,101,device=device)
    y=torch.randn(8,101,device=device)
    risk=torch.ones_like(y)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        scores,aux=model(x,mask)
        loss=robust_objective(scores,y,mask,'think_mix_huber',aux,risk)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    start=time.time()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        scores,aux=model(x,mask)
        loss=robust_objective(scores,y,mask,'think_mix_huber',aux,risk)
        loss.backward()
        optimizer.step()
        float(loss.detach())
    torch.cuda.synchronize()
    return {'seed':seed,'seconds':time.time()-start,'allocated_MB':torch.cuda.max_memory_allocated()/1e6,'free_MB':torch.cuda.mem_get_info()[0]/1e6}

if __name__=='__main__':
    for workers in [1,3]:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            print(workers,list(pool.map(measure,range(workers))),flush=True)
