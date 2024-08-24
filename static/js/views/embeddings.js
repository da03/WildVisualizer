function isMobileDevice() {
    //return 1==1;
    return (typeof window.orientation !== "undefined") || (navigator.userAgent.indexOf('IEMobile') !== -1);
}
let deckgl = null;
let layer = null;
let highlightedPointId = null;
$(document).ready(function () {
    ['zoom-out', 'zoom-reset', 'zoom-in'].forEach(id => {
        const button = document.getElementById(id);
        button.addEventListener("touchstart", function(){ 
            this.classList.remove('mobileHoverFix');
        });
        button.addEventListener("touchend", function(){
            const btn = this;
            setTimeout(function() {
                btn.classList.add('mobileHoverFix');
            }, 150); // 150 milliseconds delay
        });
    });

    if (isMobileDevice()) {
      document.getElementById('mobile-instructions').style.display = 'block';
    } else {
      document.getElementById('desktop-instructions').style.display = 'block';
    }
    function showMobileTooltip(object, x, y) {
        const tooltipEl = document.getElementById('tooltip');
        if (object && tooltipEl) {
            const tooltipContent = `
                <div><strong>${escapeHTML(object.dataset)}</strong></div>
                <div class="chat-container">
                    <div class="messages mine" style="max-height: 40vh; overflow-y: auto;">
                        <div class="message last">${escapeHTML(object.c)}</div>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px;">
                <button class="btn btn-outline-secondary btn-sm close-tooltip" onclick="event.stopPropagation(); this.closest('#tooltip').style.display='none'; highlightedPointId = null; layer = layer.clone({updateTriggers: {getFillColor: [highlightedPointId], getRadius: [highlightedPointId], getLineWidth: [highlightedPointId], getPosition: [highlightedPointId]}}); deckgl.setProps({layers: [layer]});">
                    <i class="fas fa-times"></i>
                </button>
                <a target="_blank" href="/conversation/${object.dataset}/${object.i}" class="btn btn-outline-secondary btn-sm" style="display: inline-block;">View Full</a>
            </div>
                </div>
            `;
            tooltipEl.innerHTML = tooltipContent;
            tooltipEl.style.width = '44%';
            //tooltipEl.style.maxHeight = '60vh';
            tooltipEl.style.overflowY = 'auto';
            tooltipEl.style.overflowY = 'auto';
            tooltipEl.style.zIndex = '1500';
            //tooltipEl.style.display = 'flex';
            tooltipEl.style.display = 'flex';
            tooltipEl.style.flexDirection = 'column';
            //tooltipEl.style.left = `${x}px`;
            //tooltipEl.style.top = `${y}px`;
            // Close tooltip when clicking outside
            //document.addEventListener('click', closeMobileTooltip);
        } else if (tooltipEl) {
            tooltipEl.style.display = 'none';
        }
    
    }
    
    function closeMobileTooltip(event) {
        const tooltipEl = document.getElementById('tooltip');
        if (tooltipEl && !tooltipEl.contains(event.target)) {
            tooltipEl.style.display = 'none';
            //document.removeEventListener('click', closeMobileTooltip);
        }
    }

    function updateTooltip(object, x, y) {
        const tooltipEl = document.getElementById('tooltip');
        if (object && tooltipEl) {
            tooltipEl.style.pointerEvents = 'none';
            const tooltipContent = `
              <div><strong>${escapeHTML(object.dataset)}</strong></div>
              <div class="chat-container">
                <div class="messages mine">
                  <div class="message last">${escapeHTML(object.c)}</div>
                </div>
              </div>`;
            tooltipEl.innerHTML = tooltipContent;
            //tooltipEl.style.top = `${y}px`;
            //tooltipEl.style.left = `${x}px`;
            tooltipEl.style.display = 'block';
        } else if (tooltipEl) {
            tooltipEl.style.display = 'none';
        }
    }

    function escapeHTML(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
    }
    function showLoading() {
        document.getElementById('loading-overlay').style.display = 'block';
    }
    
    function hideLoading() {
        document.getElementById('loading-overlay').style.display = 'none';
    }
    const visualizationLanguage = window.visualizationLanguage;
    const INITIAL_VIEW_STATE = {
        target: [0, 0, 0],
        longitude: 0,
        latitude: 0,
        zoom: 0,
        minZoom: -10,
        maxZoom: 10,
        pitch: 0,
        bearing: 0,
    };

    let currentViewState = INITIAL_VIEW_STATE;
    deckgl = new deck.DeckGL({
        container: 'scatter-plot',
        map: false,
        views: new deck.OrthographicView({controller: true, width: '100%', height: '100%'}),
        viewState: INITIAL_VIEW_STATE,
        controller: true,
        layers: [],
        onViewStateChange: e => {
            //console.log('changed');
            currentViewState = e.viewState;
            deckgl.setProps({
              viewState: currentViewState
            });
        },
        getCursor: ({isDragging, isHovering}) => {
            if (isDragging) {
              return 'grabbing';
            } else if (isHovering) {
              return 'pointer';
            } else {
              return 'default';
            }
        },
    });

    let allData = [];

    const fetchData = (dataset) => {
		const file = `/static/${visualizationLanguage}/${dataset}_embeddings.json`;
        return fetch(file)
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Network response was not ok ' + response.statusText);
                }
                return response.json();
            });
    };

    const normalizeData = (data) => {
        const getPercentile = (arr, p) => {
            const sorted = [...arr].sort((a, b) => a - b);
            const index = Math.floor(sorted.length * p);
            return sorted[index];
        };

        const xValues = data.map(d => d.e[0]);
        const yValues = data.map(d => d.e[1]);

        const medianX = getPercentile(xValues, 0.5);
        const medianY = getPercentile(yValues, 0.5);
        const q1X = getPercentile(xValues, 0.25);
        const q3X = getPercentile(xValues, 0.75);
        const q1Y = getPercentile(yValues, 0.25);
        const q3Y = getPercentile(yValues, 0.75);
        const iqrX = q3X - q1X;
        const iqrY = q3Y - q1Y;

        // Use 1.5 * IQR for clamping (a common statistical practice)
        //const clampLowerX = medianX - 1.5 * iqrX;
        //const clampUpperX = medianX + 1.5 * iqrX;
        //const clampLowerY = medianY - 1.5 * iqrY;
        //const clampUpperY = medianY + 1.5 * iqrY;

        const clampLowerX = medianX - 1.75 * iqrX;
        const clampUpperX = medianX + 1.75 * iqrX;
        const clampLowerY = medianY - 1.75 * iqrY;
        const clampUpperY = medianY + 1.75 * iqrY;

        const plot_size = () => {
            const cont = document.getElementById("scatter-plot");
            const wh = cont.offsetHeight - 5*2;
            const ww = cont.offsetWidth - 5*2;
            return [ww, wh];
        };

        const [pW, pH] = plot_size();
        const scaleX = pW / (clampUpperX - clampLowerX);
        const scaleY = pH / (clampUpperY - clampLowerY);
        const scale = Math.min(scaleX, scaleY);
    
        const clamp = (value, min, max) => Math.min(Math.max(value, min), max);


        return data.map(d => ({
            position: [
                (d.e[0] - medianX) * scaleX, (d.e[1] - medianY) * scaleY
                //((d.e[0] - minX) / (maxX - minX)) * pW - pW/2,
                //((d.e[1] - minY) / (maxY - minY)) * pH - pH/2
            ],
            i: d.i,
            dataset: d.dataset,
            c: d.c
        }));
    };

    // Update view state
    const updateLayer = (data, highlightIds = []) => {
        const normalizedData = normalizeData(data);
        layer = new deck.ScatterplotLayer({
            coordinateSystem: deck.COORDINATE_SYSTEM.CARTESIAN,
            coordinateOrigin: [0, 0, 0],
            id: 'scatterplot-layer',
            data: normalizedData,
            pickable: true,
            opacity: 0.8,
            stroked: true,
            filled: true,
            radiusScale: 1,
            radiusMinPixels: 1,
            radiusMaxPixels: 6,
            lineWidthUnits: 'pixels',
            lineWidthMinPixels: 0,
            lineWidthMaxPixels: 2,
            //getPosition: (d) => d.position,
            getPosition: (d) => {
                let z = 0;
                if (d.i === highlightedPointId) {
                    z = 0.002; // Highest z-index for the highlighted point
                } else if (highlightIds.includes(String(d.i))) {
                    z = 0.001; // Slightly raised for filtered points
                }
                //const z = highlightIds.includes(String(d.i)) ? 0.001 : 0;
                return [...d.position, z];
            },
            //getRadius: 3,
            getRadius: d => (d.i === highlightedPointId) ? 6 : 3, // Increase size for highlighted point
            //getLineWidth: 1,
            getLineWidth: d => (d.i === highlightedPointId) ? 2 : 1, // Thicker outline for highlighted point
            getFillColor: (d) => {
                if (d.i === highlightedPointId) {
                    return [255, 255, 0]; // Yellow for the highlighted point
                } else if (highlightIds.includes(String(d.i))) {
                    return [255, 0, 0]; // Red for highlighted
                }
                return d.dataset === 'wildchat' ? [0, 255, 0] : [0, 0, 255]; // Green for wildchat, Blue for lmsyschat
                //if (highlightIds.includes(String(d.i))) {
                //  return [255, 0, 0]; // Red for highlighted
                //}
                //return d.dataset === 'wildchat' ? [0, 255, 0] : [0, 0, 255]; // Green for wildchat, Blue for lmsyschat
            },
            getLineColor: [0, 0, 0],
            onHover: ({object, x, y}) => {
                if (!isMobileDevice()) {
                    const newHighlightedPointId = object ? object.i : null;
                    if (newHighlightedPointId != highlightedPointId) {
                        highlightedPointId = newHighlightedPointId;
                        layer = layer.clone({updateTriggers: {getFillColor: [highlightedPointId], getRadius: [highlightedPointId], getLineWidth: [highlightedPointId], getPosition: [highlightedPointId]}});
                        deckgl.setProps({layers: [layer]});
                    }
                    updateTooltip(object, x, y);
                } //else {
                    //showMobileTooltip(object, x, y);
                //}
            },
            updateTriggers: {
                getFillColor: [highlightedPointId],
                getRadius: [highlightedPointId],
                getLineWidth: [highlightedPointId],
                getPosition: [highlightedPointId]
            },
            onClick: ({object, x, y}) => {
                if (object) {
                    if (isMobileDevice()) {
                        const newHighlightedPointId = object.i;
                        if (newHighlightedPointId != highlightedPointId) {
                            highlightedPointId = newHighlightedPointId;
                            layer = layer.clone({updateTriggers: {getFillColor: [highlightedPointId], getRadius: [highlightedPointId], getLineWidth: [highlightedPointId], getPosition: [highlightedPointId]}});
                            deckgl.setProps({layers: [layer]});
                        }
                        showMobileTooltip(object, x, y);
                        //layer.setNeedsRedraw(true);
                        //updateTooltip(object, x, y);
                    } else {
                        window.open(`/conversation/${object.dataset}/${object.i}`, '_blank');
                    }
                } else {
                    const newHighlightedPointId = null;
                    if (newHighlightedPointId != highlightedPointId) {
                        highlightedPointId = newHighlightedPointId;
                        layer = layer.clone({updateTriggers: {getFillColor: [highlightedPointId], getRadius: [highlightedPointId], getLineWidth: [highlightedPointId], getPosition: [highlightedPointId]}});
                        deckgl.setProps({layers: [layer]});
                    }
                    closeMobileTooltip();
                    //layer.setNeedsRedraw(true);
                }
            },
        });
        deckgl.setProps({layers: [layer]});
    };

    // Function to update the displayed current filters
    const updateCurrentFilters = () => {
        const currentFiltersEl = $('#current-filters ul');
        currentFiltersEl.empty(); // Clear existing filters

        let urlParams = new URLSearchParams(window.location.search);

        if (urlParams.toString()) {
            for (const [key, value] of urlParams.entries()) {
                if (key !== 'page') {
                    currentFiltersEl.append(`
                      <li class="list-inline-item">
                        <span class="badge badge-primary">
                          ${key === 'redacted' ? 'Contains personal info' : key.charAt(0).toUpperCase() + key.slice(1)}: ${value}
                          <a href="javascript:void(0)" class="remove-filter btn btn-outline-secondary" data-filter="${key}">&times;</a>
                        </span>
                      </li>
                    `);
                }
            }
        } else {
            currentFiltersEl.append(`
              <li class="list-inline-item">
                <span class="badge badge-secondary">None</span>
              </li>
            `);
        }

        // Re-bind remove filter event
        $('.remove-filter').click(removeFilter);
    };


  const applyFilters = () => {
      showLoading();
      let urlParams = new URLSearchParams(window.location.search);
      const filters = {
          contains: $('#search-input').val(),
          toxic: $('#filter-toxic').val(),
          hashed_ip: $('#filter-hashed-ip').val(),
          language: $('#filter-language').val(),
          country: $('#filter-country').val(),
          state: $('#filter-state').val(),
          min_turns: $('#filter-min-turns').val(),
          model: $('#filter-model').val(),
          redacted: $('#filter-redacted').val(),
          dataset: $('#filter-dataset').val(),
          conversation_id: ''
      };
      // Update URL parameters
      for (const [key, value] of Object.entries(filters)) {
          if (value) {
              urlParams.set(key, value);
          } else {
              urlParams.delete(key);
          }
      }
      const queryString = urlParams.toString();
      const newUrl = queryString ? "?" + queryString : window.location.pathname;
      window.history.replaceState(null, null, newUrl);
      updateCurrentFilters();

      const newFilters = {
          ...filters, // Copy all existing key-value pairs from filters
          visualization_language: visualizationLanguage
      };


      fetch('/search_embeddings', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(newFilters)
      })
      .then((response) => response.json())
      .then((filteredEmbeddings) => {
        const highlightIds = Object.keys(filteredEmbeddings);
        //console.log(highlightIds);
        const filteredData = highlightIds.map(id => ({
          e: filteredEmbeddings[id].e,
          i: filteredEmbeddings[id].i,
          dataset: filteredEmbeddings[id].d,
          c: filteredEmbeddings[id].c
        }));

        // Merge new data with existing data while avoiding duplicates
        //filteredData.forEach(d => {
        //  if (!allData.some(nd => nd.i === d.i)) {
        //    allData.push(d);
        //  }
        //});
        const mergedData = [...allData, ...filteredData].reduce((acc, d) => {
            // Use d.i as the key in the Map
            acc.set(d.i, d);
            return acc;
        }, new Map());
        const uniqueDataArray = Array.from(mergedData.values());


        updateLayer(uniqueDataArray, highlightIds);
        hideLoading();
      })
      .catch((error) => {
        console.error('Error fetching filtered embeddings:', error);
        hideLoading();
      });
    };

    function checkInputs() {
        $('.filter-input').each(function () {
            let $input = $(this);
            let $button = $input.parent().find('.add-filter');
            if ($input.val().trim() === '') {
                $button.prop('disabled', true);
            } else {
                $button.prop('disabled', false);
            }
        });
    }
    checkInputs();
    const removeFilter = function () {
        let urlParams = new URLSearchParams(window.location.search);
        let filter = $(this).data('filter');
        if (filter == 'contains') {
            $('#search-input').val('');
        } else if (filter == 'toxic') {
            $('#filter-toxic').val('');
        } else if (filter == 'redacted') {
            $('#filter-redacted').val('');
        } else if (filter == 'dataset') {
            $('#filter-dataset').val('');
        } else if (filter == 'model') {
            $('#filter-model').val('');
        } else if (filter == 'hashed_ip') {
            $('#filter-hashed-ip').val('');
        } else if (filter == 'language') {
            $('#filter-language').val('');
        } else if (filter == 'country') {
            $('#filter-country').val('');
        } else if (filter == 'state') {
            $('#filter-state').val('');
        } else if (filter == 'min_turns') {
            $('#filter-min-turns').val('');
        }
        checkInputs();
        urlParams.delete(filter);
        const queryString = urlParams.toString();
        const newUrl = queryString ? "?" + queryString : window.location.pathname;
        window.history.replaceState(null, null, newUrl);
        updateCurrentFilters();
        applyFilters();
    }
    $('.remove-filter').click(removeFilter);
    // Fetch data for both datasets
    //Promise.all([fetchData('/static/wildchat_embeddings.json'), fetchData('/static/lmsyschat_embeddings.json')])
    //showLoading();
    Promise.all([fetchData('wildchat'), fetchData('lmsyschat')])
        .then(([wildchatData, lmsyschatData]) => {
            //console.log('Data loaded:', {wildchatData, lmsyschatData});
            wildchatData.forEach(d => d.dataset = 'wildchat');
            lmsyschatData.forEach(d => d.dataset = 'lmsyschat');
            // Combine and normalize the data
            allData = [...wildchatData, ...lmsyschatData];
            updateLayer(allData);
            applyFilters();

            function handleEnterKey(event) {
              if (event.key === 'Enter') {
                applyFilters();
              }
            }
            $('.filter-input').on('input keyup change', checkInputs);
            $('.add-filter').click(applyFilters);
            $('.filter-input').on('keypress', handleEnterKey);

            // Zoom controls
            $('#zoom-in').on('click', () => {
              currentViewState = {...currentViewState, zoom: currentViewState.zoom+0.5}
              deckgl.setProps({viewState: currentViewState});
            });

            $('#zoom-out').on('click', () => {
              currentViewState = {...currentViewState, zoom: currentViewState.zoom-0.5}
              deckgl.setProps({viewState: currentViewState});
            });
            $('#zoom-reset').on('click', () => {
              currentViewState = {...currentViewState, zoom: 0, target: [0, 0, 0]}
              deckgl.setProps({viewState: currentViewState});
            });
            //hideLoading();
        })
        .catch((error) => {
            console.error('Error loading data:', error);
            hideLoading();
        });
    // Add tooltip div to HTML
    const tooltipDiv = document.createElement('div');
    tooltipDiv.id = 'tooltip';
    tooltipDiv.style.position = 'fixed';
    tooltipDiv.style.width = '33%';
    //tooltipDiv.style.height = '441px';
    tooltipDiv.style.top = '10px';
    tooltipDiv.style.right = '10px';
    tooltipDiv.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
    //tooltipDiv.style.backgroundColor = 'transparent';
    tooltipDiv.style.color = 'black';
    tooltipDiv.style.padding = '5px';
    tooltipDiv.style.border = '1px solid #ccc';
    tooltipDiv.style.borderRadius = '3px';
    tooltipDiv.style.overflowY = 'auto';
    tooltipDiv.style.display = 'none';
    document.body.appendChild(tooltipDiv);
    //deckgl.canvas.addEventListener('mouseleave', hideTooltip);
});

