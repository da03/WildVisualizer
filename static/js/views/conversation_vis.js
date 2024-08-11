$(document).ready(function () {
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
    const deckgl = new deck.DeckGL({
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

    const fetchData = (file) => {
        return fetch(file)
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Network response was not ok ' + response.statusText);
                }
                return response.json();
            });
    };

    const normalizeData = (data) => {
        const minX = Math.min(...data.map(d => d.e[0]));
        const maxX = Math.max(...data.map(d => d.e[0]));
        const minY = Math.min(...data.map(d => d.e[1]));
        const maxY = Math.max(...data.map(d => d.e[1]));

        const plot_size = () => {
            const cont = document.getElementById("scatter-plot");
            const wh = cont.offsetHeight - 5*2;
            const ww = cont.offsetWidth - 5*2;
            return [ww, wh];
        };

        const [pW, pH] = plot_size();

        return data.map(d => ({
            position: [
                ((d.e[0] - minX) / (maxX - minX)) * pW - pW/2,
                ((d.e[1] - minY) / (maxY - minY)) * pH - pH/2
            ],
            i: d.i,
            dataset: d.dataset,
            c: d.c
        }));
    };

    // Update view state
    const updateLayer = (data, highlightIds = []) => {
        const normalizedData = normalizeData(data);
        const layer = new deck.ScatterplotLayer({
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
            lineWidthMinPixels: 1,
            lineWidthMaxPixels: 2,
            getPosition: (d) => d.position,
            getRadius: 3,
            getFillColor: (d) => {
                if (highlightIds.includes(String(d.i))) {
                  return [255, 0, 0]; // Red for highlighted
                }
                return d.dataset === 'wildchat' ? [0, 255, 0] : [0, 0, 255]; // Green for wildchat, Blue for lmsyschat
            },
            getLineColor: [0, 0, 0],
            onHover: ({object, x, y}) => {
                const tooltipEl = document.getElementById('tooltip');
                if (object && tooltipEl) {
                    const tooltipContent = `
                      <div><strong>${object.dataset}</strong></div>
                      <div class="chat-container">
                        <div class="messages mine">
                          <div class="message last">${object.c}</div>
                        </div>
                      </div>`;
                    tooltipEl.innerHTML = tooltipContent;
                    //tooltipEl.style.top = `${y}px`;
                    //tooltipEl.style.left = `${x}px`;
                    tooltipEl.style.display = 'block';
                } else if (tooltipEl) {
                    tooltipEl.style.display = 'none';
                }
            },
            onClick: ({object}) => {
                if (object) {
                    //console.log('Clicked on:', object);
                    window.open(`/conversation/${object.dataset}/${object.i}`, '_blank');
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
      window.history.replaceState(null, null, "?" + urlParams.toString());
      updateCurrentFilters();

      fetch('/search_embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(filters)
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
        filteredData.forEach(d => {
          if (!allData.some(nd => nd.i === d.i)) {
            allData.push(d);
          }
        });

        updateLayer(allData, highlightIds);
      })
      .catch((error) => {
        console.error('Error fetching filtered embeddings:', error);
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
        window.history.replaceState(null, null, "?" + urlParams.toString());
        updateCurrentFilters();
        applyFilters();
    }
    $('.remove-filter').click(removeFilter);
    // Fetch data for both datasets
    Promise.all([fetchData('/static/wildchat_embeddings.json'), fetchData('/static/lmsyschat_embeddings.json')])
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
        })
        .catch((error) => {
            console.error('Error loading data:', error);
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
    tooltipDiv.style.pointerEvents = 'none';
    tooltipDiv.style.overflowY = 'auto';
    tooltipDiv.style.display = 'none';
    document.body.appendChild(tooltipDiv);
});

