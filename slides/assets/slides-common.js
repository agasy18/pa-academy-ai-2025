(function (global) {
  'use strict';

  function mergePlugins(targetPlugins, fallbackPlugins) {
    if (!Array.isArray(targetPlugins) || targetPlugins.length === 0) {
      return fallbackPlugins.slice();
    }
    const merged = targetPlugins.slice();
    fallbackPlugins.forEach((plugin) => {
      if (plugin && merged.indexOf(plugin) === -1) {
        merged.push(plugin);
      }
    });
    return merged;
  }

  function renderGraphvizBlocks() {
    if (typeof global.Viz === 'undefined') {
      return;
    }

    const viz = new global.Viz();
    const renderBlocks = () => {
      const blocks = global.document.querySelectorAll('code.language-dot[data-graphviz]');
      blocks.forEach((code) => {
        const dot = code.textContent;
        viz.renderSVGElement(dot)
          .then((svg) => {
            const container = global.document.createElement('div');
            container.className = 'graphviz';
            container.appendChild(svg);

            const link = global.document.createElement('a');
            link.href = '#';
            link.className = 'graphviz-fullscreen-link';
            link.innerHTML = '&#x26F6;'; // square with diagonal arrow
            link.title = 'Open diagram fullscreen';
            link.addEventListener('click', (event) => {
              event.preventDefault();
              event.stopPropagation();
              const popup = global.open('', '_blank');
              if (!popup || popup.closed) {
                return;
              }
              const svgMarkup = svg.outerHTML;
              popup.document.write(
                '<!doctype html><html><head>' +
                  '<meta charset="utf-8" />' +
                  '<title>Diagram</title>' +
                  '<style>' +
                  'html, body { margin: 0; padding: 0; height: 100%; background: #ffffff; }' +
                  'body { display: flex; align-items: center; justify-content: center; }' +
                  'svg { max-width: 95vw; max-height: 95vh; width: auto; height: auto; }' +
                  '</style>' +
                '</head><body>' +
                  svgMarkup +
                '</body></html>',
              );
              popup.document.close();
            });

            const wrapper = global.document.createElement('div');
            wrapper.className = 'graphviz-wrapper';
            container.appendChild(link);
            wrapper.appendChild(container);
            const pre = code.parentNode;
            if (pre && pre.parentNode) {
              pre.parentNode.replaceChild(wrapper, pre);
            }
          })
          .catch((err) => {
            // eslint-disable-next-line no-console
            console.error('Graphviz render error:', err);
          });
      });
    };

    if (global.document.readyState === 'loading') {
      global.document.addEventListener('DOMContentLoaded', renderBlocks, { once: true });
    } else {
      renderBlocks();
    }
  }

  function initializeSlides(options) {
    if (!global.Reveal || typeof global.Reveal.initialize !== 'function') {
      throw new Error('Reveal.js is required before calling initializeSlides');
    }

    const {
      graphviz = true,
      plugins = [],
      ...revealOptions
    } = options || {};

    const defaultPlugins = [];
    if (typeof global.RevealNotes !== 'undefined') {
      defaultPlugins.push(global.RevealNotes);
    }
    if (typeof global.RevealHighlight !== 'undefined') {
      defaultPlugins.push(global.RevealHighlight);
    }

    const config = {
      hash: true,
      slideNumber: true,
      transition: 'slide',
      ...revealOptions,
      plugins: mergePlugins(plugins, defaultPlugins),
    };

    global.Reveal.initialize(config);

    if (graphviz) {
      renderGraphvizBlocks();
    }
  }

  global.initializeSlides = initializeSlides;
})(window);
