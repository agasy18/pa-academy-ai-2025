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
            const pre = code.parentNode;
            if (pre && pre.parentNode) {
              pre.parentNode.replaceChild(container, pre);
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

