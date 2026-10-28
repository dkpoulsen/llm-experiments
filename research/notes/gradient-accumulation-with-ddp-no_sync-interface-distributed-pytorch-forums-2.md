---
title: Gradient accumulation with DDP no_sync interface - distributed - PyTorch Forums
id: gradient-accumulation-with-ddp-no_sync-interface-distributed-pytorch-forums-2
tags:
- cmp170hx-training-feasibility-a0d4fc
- ddp
- no-sync
- gradient-accumulation
- pytorch-forum
- sub-q4
created: '2026-10-27T10:14:36.149088Z'
updated: '2026-10-27T19:00:31.189621Z'
source: https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593
source_domain: discuss.pytorch.org
fetched_at: '2026-10-27T10:14:35.510949Z'
fetch_provider: crawl4ai
status: review
type: note
tier: practitioner
content_type: forum
deprecated: false
summary: 'PyTorch Forums thread (Jan 2023 - Jun 2024) on using DDP''s no_sync() for
  gradient accumulation. A user accumulates 4 micro-batches of 64 to emulate batch
  256 and observes a lower loss than a true 256 batch, suspecting hidden synchronization.
  Maintainer rvarm1 replies that the no_sync usage ''looks like correct usage'' and
  warns optimizer.step() should run after gradients are allreduced or ranks desynchronize;
  a later reply (gilg) identifies the real bug - the snippet calls optimizer.zero_grad()
  and optimizer.step() on the sync iteration in a way that uses only that micro-batch''s
  gradient, and accumulated gradients must be normalized by 1/4. Takeaway for this
  run: no_sync suppresses the per-micro-batch AllReduce but does not change gradient
  semantics - loss normalization and step placement stay the user''s responsibility.'
utility_score: 12.0
---

[ Skip to last reply ](https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593/5) [ Skip to top ](https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593/1)
[ Skip to main content ](https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593#main-container)
#  [ Gradient accumulation with DDP no_sync interface ](https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593)
You have selected **0** posts.
[ select all ](https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593)
[ cancel selecting ](https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593)
3.8k views 
[ Jan 2023  ](https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593/1 "Jump to the first post")
1 / 5 
Jan 2023 
[ Jun 2024 ](https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593/5)
##  post by caesar025 on Jan 4, 2023 
[ caesar025 ](https://discuss.pytorch.org/u/caesar025)
[ Jan 2023 ](https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593 "Post date")
I am currently trying to perform gradient accumulation in a DistributedDataParallel setting for simulating large batch sizes on a small number of GPUs. As far as I know, that should be possible by skipping the AllReduce operation.
Horovod features the _backward_passes_per_step_ option for such cases - I read that the same should be possible in PyTorch-DDP with _no_sync_ interface.
My original batch size is 64, and I try to “simulate” a batch size that is 4x as large (=256).
This is a snippet of my current code:

```
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        if (i % 4 == 0):
        # perform regular AllReduce operation every four steps 
            output = model(images)
            loss = criterion(output, target)
            train_loss += loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
        # accumulate gradients using the no_sync interface
           with model.no_sync():
                   output = model(images)
                   loss = criterion(output, target)
                   train_loss += loss 

                   # skipping the optimizer call here 
                   loss.backward()


```

However, when comparing the training loss of this “simulated” large batch run (with gradient accumulation) with actually running with batch size 256, I get a **lower** loss for the gradient accumulation run than for the run with the actual batch size 256.
I suspect there are some hidden synchronization steps in my code that occur despite the use of the _no_sync_ interface (and therefore get me a better training loss)? Is that the case here and how could that be fixed?
3.8k views 
##  post by rvarm1 on Jan 10, 2023 
[ rvarm1 ](https://discuss.pytorch.org/u/rvarm1)
[ Jan 2023 ](https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593/2 "Post date")
This looks like correct usage. With regard to the loss difference, have you checked how the gradients differ between no_sync versus larger batch size runs, and ensured that your training is as deterministic as possible?
Also, I think as typical practice, optimizer.step() should be done after gradients are allreduced, otherwise the model may become out of sync across ranks.
1 year later 
##  post by gilg on Jun 12, 2024 
[ Jun 2024 ](https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593/3 "Post date")
This is an old question, but for the sake of other readers, I believe there is a bug present. You execute `optimizer.zero_grad()` followed by `loss.backward()`, and then `optimizer.step()`. This means that you are using the gradient every four batches instead of accumulating it. Additionally, you should normalize the loss by a factor of 1/4 since the gradient is summed.
##  post by caesar025 on Jun 13, 2024 
##  post by gilg on Jun 13, 2024 
Reply
###  New & Unread Topics   
Topic list, column headers with buttons are sortable.  
|  Topic   |  Replies   |  Views   |  Activity   |  
| --- | --- | --- | --- |  
|  [Fix gradient accumulation for DDP with static graph and no sync](https://discuss.pytorch.org/t/fix-gradient-accumulation-for-ddp-with-static-graph-and-no-sync/225166)  |  
|   |  
|  [Is torch Muon optimizer compatible with FSDP/HSDP?](https://discuss.pytorch.org/t/is-torch-muon-optimizer-compatible-with-fsdp-hsdp/224467)  |  
|  [PyTorch Distributed (Gloo) fails with system error: 10049 - The requested address is not valid in its context](https://discuss.pytorch.org/t/pytorch-distributed-gloo-fails-with-system-error-10049-the-requested-address-is-not-valid-in-its-context/224907)  |  
|  [TBCCL: Custom PyTorch Distributed Backend for Mac/Linux over Thunderbolt](https://discuss.pytorch.org/t/tbccl-custom-pytorch-distributed-backend-for-mac-linux-over-thunderbolt/225551)  |  
###  Want to read more? Browse other topics in [distributed](https://discuss.pytorch.org/c/distributed/12) or [view latest topics](https://discuss.pytorch.org/latest). 
[ Powered by Discourse ](https://discourse.org/powered-by)
