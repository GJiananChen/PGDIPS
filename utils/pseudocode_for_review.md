## pseudo code for PGDIPS algorithm
```
image = Load(target_image)

For epochs in max_epochs:
    loss = 0
    
    components = DeepImagePrior_models(image)
    reconstructed_image = Modified_Inverse_BeerLambert(components)
    
    loss += MeanSquaredLoss(img, reconstructed_image)
    loss += ExclusionLoss(components['concentration_maps'])
    if epoch < keep_color_epochs:
        loss += ColorFixingLoss(components['color_vectors'], default_color_vectors)
        
    update(DeepImagePrior_models, loss)
    update(Modified_Inverse_BeerLambert, loss)
    
    save_intermediate_images()
    
save_final_results()
```        
