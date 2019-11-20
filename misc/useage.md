
### DataParallel
device_ids = range(T.cuda.device_count())
model = nn.DataParallel(model, device_ids)