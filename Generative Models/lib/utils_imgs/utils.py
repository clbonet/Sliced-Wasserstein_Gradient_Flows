import torch
import matplotlib.pyplot as plt

def uniform_quantization(img):
    return (img*255+torch.rand(img.size()))/256

def rescale_logit(img,lambd=1e-6):
    ## logit space
    return torch.logit(lambd+(1-2*lambd)*img)

def inverse_logit(img,lambd=1e-6):
    return (torch.sigmoid(img)-lambd)/(1-2*lambd)


def val_mnist(model, device, dim_noise=100):
    model.eval()

    d = 28*28
    torch.manual_seed(42)
    r,c = 5,5
    z_random = torch.randn(r,c,d,device=device)

    if model.__class__.__name__ == "NormalizingFlows":
        zs, _ = model(z_random.reshape(-1,28*28))
        gen_imgs = inverse_logit(zs[-1].view(-1,28,28).detach().cpu())
    elif model.__class__.__name__ == "CNN":
        zs = model(z_random.reshape(-1,dim_noise,1,1))
        gen_imgs = zs.view(-1,28,28).detach().cpu()
    else:
        z =  model(z_random.reshape(-1,28*28))
        gen_imgs = inverse_logit(z.view(-1,28,28).detach().cpu())

    cpt = 0
    fig,ax = plt.subplots(r,c)
    for i in range(r):
        for j in range(c):
            ax[i,j].imshow(gen_imgs[cpt],"gray")
            ax[i,j].axis('off')

            cpt += 1
                
    fig.set_size_inches(6, 6)
    plt.tight_layout()
    plt.show()
    
    
def val_mnist_ae(rho, autoencoder, device, d=48):
    with torch.no_grad():
        autoencoder.eval()
        rho.eval()

        torch.manual_seed(42)
        r,c = 5,5
        z_random = torch.randn(r,c,d,device=device)

        if rho.__class__.__name__ == "NormalizingFlows":
            zs, _ = rho(z_random.reshape(-1,d))
            gen_imgs = autoencoder.decoder(zs[-1]).detach().cpu().reshape(-1,28,28)
        else:
            zs = rho(z_random.reshape(-1,d))
            gen_imgs = autoencoder.decoder(zs).detach().cpu().reshape(-1,28,28)

        cpt = 0
        fig,ax = plt.subplots(r,c)
        for i in range(r):
            for j in range(c):
                ax[i,j].imshow(gen_imgs[cpt],"gray")
                ax[i,j].axis("off")
                cpt += 1
                
        fig.set_size_inches(6, 6)
        plt.tight_layout()
        plt.show()
