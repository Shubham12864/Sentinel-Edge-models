# Deploying the landing page (GitHub Pages)

The storefront lives at `docs/index.html`. GitHub Pages can serve it directly
from this branch — no build step, no dependencies.

## One-time setup (repo admin)

1. Open **https://github.com/Shubham12864/Sentinel-Edge-models/settings/pages**
2. Under **Build and deployment**:
   - Source: **Deploy from a branch**
   - Branch: **master** · Folder: **/docs**
3. Click **Save**. First build takes ~1 minute.

Site URL: `https://shubham12864.github.io/Sentinel-Edge-models/`

## Updating the page

Edit `docs/index.html`, then:

```bash
git add docs/index.html
git commit -m "docs: update landing page"
git push origin master
```

Pages rebuilds automatically on every push to master.

## Custom domain (optional)

1. Add a `CNAME` file inside `docs/` containing your domain, e.g.:
   ```
   sentinel.yourdomain.com
   ```
2. At your DNS provider create: `CNAME sentinel → shubham12864.github.io`
3. In Settings → Pages enable **Enforce HTTPS** once the certificate issues.
