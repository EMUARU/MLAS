<template>
  <div class="about">
    <h1>{{ $t('lblSynthetic') }}</h1>
    <h6 class="text-body-secondary">{{ $t('msgSyntheticDescription') }}</h6>
  </div>
  
  <!-- 橫線 -->
  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <form class="row g-3 needs-validation" @submit.prevent="runGenerate" style="margin-top: 16px">
    <!-- 生成器類型 -->
    <div class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblGeneratorType') }}</label>
      <div class="col-sm-8">
        <select class="form-select" v-model="selected.generator_type" :disabled="loading">
          <option value="ctgan">CTGAN - Conditional Tabular GAN</option>
          <option value="tvae">TVAE - Tabular VAE</option>
          <option value="tabddpm">TabDDPM - Tabular Diffusion Model</option>
        </select>
        <div class="form-text">{{ generatorDescription }}</div>
        <div v-if="errors.generator_type" class="text-danger small">{{ errors.generator_type }}</div>
      </div>
    </div>

    <!-- 表格式資料 -->
    <div class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblTabularData') }}</label>
      <div class="col-sm-7">
        <select class="form-select" v-model="selected.data" :disabled="loading">
          <option value="">{{ $t('lblPleaseSelect') }}</option>
          <option v-for="file in files" :key="file" :value="file">{{ file }}</option>
        </select>
        <div v-if="errors.data" class="text-danger small">{{ errors.data }}</div>
      </div>
      <div class="col-sm-1">
        <button v-if="preview_data.columns.length > 0" class="btn btn-outline-primary" style="white-space: nowrap" type="button" @click="toggleCollapse" :disabled="loading">{{ $t('lblPreview') }}</button>
      </div>
    </div>

    <!-- Preview -->
    <div v-if="preview_data.total_rows != 0" class="row mb-3">
      <div class="collapse" ref="collapsePreview">
        <div class="card card-body">
          <div class="table-responsive">
            <table class="table">
              <caption>{{ $t('msgPreviewCaption', { count: preview_data.total_rows }) }}</caption>
              <thead>
                <tr>
                  <th></th>
                  <th v-for="col in preview_data.columns" :key="col">
                    {{ col }}
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, rowIndex) in preview_data.preview" :key="rowIndex">
                  <td></td>
                  <td v-for="col in preview_data.columns" :key="col">
                    {{ row[col] }}
                  </td>
                </tr>
                <tr>
                  <td></td>
                  <td v-for="col in preview_data.columns" :key="col">
                    ...
                  </td>
                </tr>
              </tbody>
              <tfoot>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblMin') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'min-' + col">
                    {{ summary[col] ? summary[col].min : '-' }}
                  </td>
                </tr>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblMax') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'max-' + col">
                    {{ summary[col] ? summary[col].max : '-' }}
                  </td>
                </tr>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblMedian') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'median-' + col">
                    {{ summary[col] ? summary[col].median : '-' }}
                  </td>
                </tr>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblMean') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'mean-' + col">
                    {{ summary[col] && summary[col].mean !== null ? summary[col].mean.toFixed(2) : '-' }}
                  </td>
                </tr>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblMode') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'mode-' + col">
                    {{ summary[col] ? summary[col].mode : '-' }}
                  </td>
                </tr>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblStd') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'std-' + col">
                    {{ summary[col] && summary[col].std !== null ? summary[col].std.toFixed(2) : '-' }}
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- 標籤欄位（單選 - 僅能選擇一個） -->
    <div v-if="preview_data.total_rows != 0" class="row mb-3">
      <label class="col-sm-3 col-form-label">
        {{ $t('lblLabelColumn') }}
        <span class="text-muted small">({{ $t('lblOptional') }})</span>
      </label>
      <div class="col-sm-8">
        <select class="form-select" v-model="selected.label_column" :disabled="loading">
          <option value="none">{{ $t('lblNoLabel') }}</option>
          <option v-for="col in preview_data.columns" :key="col" :value="col">{{ col }}</option>
        </select>
        <div class="form-text">{{ $t('msgLabelColumnHint') }}</div>
      </div>
    </div>

    <!-- 生成數量（TGANT 關閉時顯示） -->
    <div v-if="!selected.enable_tgant" class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblNumSamples') }}</label>
      <div class="col-sm-4">
        <select class="form-select" v-model="selected.num_samples_type" :disabled="loading">
          <option value="1x">1x ({{ $t('lblSameAsOriginal') }})</option>
          <option value="2x">2x</option>
          <option value="5x">5x</option>
          <option value="10x">10x</option>
          <option value="100x">100x</option>
          <option value="custom">{{ $t('lblCustom') }}</option>
        </select>
      </div>
      <div v-if="selected.num_samples_type === 'custom'" class="col-sm-4">
        <input v-model="selected.custom_num_samples" type="number" class="form-control" 
               :placeholder="$t('lblEnterNumber')" :disabled="loading" min="1">
        <div v-if="errors.custom_num_samples" class="text-danger small">{{ errors.custom_num_samples }}</div>
      </div>
    </div>

    <!-- TGANT 篩選（有標籤欄位時顯示） -->
    <div v-if="selected.label_column !== 'none'" class="row mb-3">
      <label class="col-sm-3 col-form-label">
        {{ $t('lblTGANTFilter') }}
        <span class="badge bg-info">{{ $t('lblCoreFeature') }}</span>
      </label>
      <div class="col-sm-8">
        <div class="form-check">
          <input class="form-check-input" type="checkbox" id="enableTGANT" 
                 v-model="selected.enable_tgant" :disabled="loading">
          <label class="form-check-label" for="enableTGANT">
            {{ $t('lblEnableTGANT') }}
            <span class="badge bg-success">{{ $t('lblRecommended') }}</span>
          </label>
        </div>
        <div v-if="selected.enable_tgant" class="mt-2">
          <label class="form-label small">{{ $t('lblGenerateMultiplier') }}</label>
          <select class="form-select form-select-sm" v-model="selected.generate_multiplier" :disabled="loading" style="width: 200px;">
            <option value="10">10x</option>
            <option value="50">50x</option>
            <option value="100">100x ({{ $t('lblRecommended') }})</option>
          </select>
          <div class="form-text text-info">
            <i class="fa fa-info-circle"></i> {{ $t('msgTGANTGenerateHint') }}
          </div>
          
          <!-- TGANT 進階參數 -->
          <div class="mt-3 p-2 bg-light rounded">
            <div class="small text-muted mb-2"><i class="fa fa-cog"></i> {{ $t('lblTGANTAdvanced') }}</div>
            
            <!-- 收斂門檻 -->
            <div class="row mb-2 align-items-center">
              <label class="col-4 small">{{ $t('lblTGANTThreshold') }}</label>
              <div class="col-8">
                <input type="number" class="form-control form-control-sm" v-model.number="selected.tgant_threshold" 
                       :disabled="loading" min="0.5" max="0.99" step="0.05" style="width: 80px;">
                <small class="text-muted ms-1">(0.5~0.99)</small>
              </div>
            </div>
            
            <!-- 最大迭代代數 -->
            <div class="row mb-2 align-items-center">
              <label class="col-4 small">{{ $t('lblTGANTMaxGen') }}</label>
              <div class="col-8">
                <input type="number" class="form-control form-control-sm" v-model.number="selected.tgant_max_gen" 
                       :disabled="loading" min="3" max="50" step="1" style="width: 80px;">
              </div>
            </div>
            
            <!-- 評估指標 -->
            <div class="row align-items-center">
              <label class="col-4 small">{{ $t('lblTGANTMetric') }}</label>
              <div class="col-8">
                <select class="form-select form-select-sm" v-model="selected.tgant_metric" :disabled="loading" style="width: 140px;">
                  <option value="min">min ({{ $t('lblStrict') }})</option>
                  <option value="macro">macro ({{ $t('lblRelaxed') }})</option>
                </select>
              </div>
            </div>
            <div class="form-text small mt-1">{{ $t('msgTGANTMetricHint') }}</div>
          </div>
        </div>
        <div class="form-text">{{ $t('msgTGANTHint') }}</div>
      </div>
    </div>

    <!-- 生成數量（TGANT 開啟時顯示提示） -->
    <div v-if="selected.enable_tgant && selected.label_column !== 'none'" class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblNumSamples') }}</label>
      <div class="col-sm-8">
        <span class="text-muted">
          <i class="fa fa-info-circle"></i> {{ $t('msgTGANTNumSamplesHint') }}
        </span>
      </div>
    </div>

    <!-- 訓練模式 -->
    <div v-if="selected.label_column !== 'none'" class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblTrainMode') }}</label>
      <div class="col-sm-8">
        <div class="form-check form-check-inline">
          <input class="form-check-input" type="radio" name="trainMode" id="trainModeMixed" 
                 value="mixed" v-model="selected.train_mode" :disabled="loading">
          <label class="form-check-label" for="trainModeMixed">{{ $t('lblTrainModeMixed') }}</label>
        </div>
        <div class="form-check form-check-inline">
          <input class="form-check-input" type="radio" name="trainMode" id="trainModeGrouped" 
                 value="grouped" v-model="selected.train_mode" :disabled="loading">
          <label class="form-check-label" for="trainModeGrouped">
            {{ $t('lblTrainModeGrouped') }}
            <span class="badge bg-success">{{ $t('lblRecommended') }}</span>
          </label>
        </div>
        <div class="form-text">{{ $t('msgTrainModeHint') }}</div>
      </div>
    </div>

    <!-- 自動參數調整 -->
    <div v-if="selected.label_column !== 'none' && selected.train_mode === 'grouped'" class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblAutoParams') }}</label>
      <div class="col-sm-8">
        <div class="form-check">
          <input class="form-check-input" type="checkbox" id="useAutoParams" 
                 v-model="selected.use_auto_params" :disabled="loading">
          <label class="form-check-label" for="useAutoParams">
            {{ $t('lblUseAutoParams') }}
            <span class="badge bg-primary">{{ $t('lblRecommended') }}</span>
          </label>
        </div>
      </div>
    </div>

    <!-- 欄位約束 (Column Constraints) -->
    <div v-if="preview_data.total_rows != 0" class="row mb-3">
      <label class="col-sm-3 col-form-label">
        {{ $t('lblColumnConstraints') }}
        <span class="text-muted small">({{ $t('lblOptional') }})</span>
      </label>
      <div class="col-sm-8">
        <button type="button" class="btn btn-outline-secondary btn-sm" @click="showConstraints = !showConstraints">
          <i :class="showConstraints ? 'fa fa-chevron-up' : 'fa fa-chevron-down'"></i>
          {{ showConstraints ? $t('lblHide') : $t('lblShow') }}
          <span v-if="(constraintRules.length + conditionalRules.length) > 0" class="badge bg-success ms-1">
            {{ constraintRules.length + conditionalRules.length }}
          </span>
        </button>
        <div class="form-text">{{ $t('msgColumnConstraintsHint') }}</div>

        <!-- 展開的約束設定區 -->
        <div v-if="showConstraints" class="mt-3">
          <!-- ⚠️ 生成模型限制說明 -->
          <div class="alert alert-info py-2 px-3 small mb-3">
            <i class="fa fa-info-circle me-1"></i>
            <strong>{{ $t('lblConstraintLimitationTitle') }}</strong>
            <div class="mt-1">{{ $t('msgConstraintLimitation') }}</div>
          </div>

          <!-- ============ 單欄位約束 ============ -->
          <h6 class="text-muted small mb-2">
            <i class="fa fa-columns"></i> {{ $t('lblSingleColumnConstraints') }}
          </h6>

          <!-- 選擇欄位 -->
          <div class="mb-3">
            <label class="form-label small">{{ $t('lblSelectColumn') }}</label>
            <select v-model="constraintSelectedColumn" class="form-select form-select-sm" 
                    :disabled="loading" @change="onConstraintColumnChange">
              <option value="">{{ $t('lblPleaseSelect') }}</option>
              <option v-for="col in preview_data.columns" :key="col" :value="col"
                      :disabled="isColumnConstrained(col)">
                {{ col }}
                <template v-if="isColumnConstrained(col)"> ({{ $t('lblAlreadySet') }})</template>
              </option>
            </select>
          </div>

          <!-- 規則設定卡片 -->
          <div v-if="constraintSelectedColumn" class="card p-3 mb-3">
            <div class="d-flex justify-content-between align-items-center mb-2">
              <div>
                <strong>{{ constraintSelectedColumn }}</strong>
                <span :class="'badge ms-2 ' + (isNumericColumn(constraintSelectedColumn) ? 'bg-secondary' : 'bg-info')">
                  {{ isNumericColumn(constraintSelectedColumn) ? $t('lblNumericColumn') : $t('lblCategoricalColumn') }}
                </span>
              </div>
            </div>

            <!-- 約束類型 -->
            <div class="mb-2">
              <label class="form-label small">{{ $t('lblConstraintType') }}</label>
              <select v-model="constraintDraft.type" class="form-select form-select-sm" :disabled="loading">
                <template v-if="isNumericColumn(constraintSelectedColumn)">
                  <option value="range">{{ $t('lblConstraintRange') }}</option>
                  <option value="min">{{ $t('lblConstraintMin') }}</option>
                  <option value="max">{{ $t('lblConstraintMax') }}</option>
                  <option value="fixed">{{ $t('lblConstraintFixed') }}</option>
                </template>
                <template v-else>
                  <option value="options">{{ $t('lblConstraintOptions') }}</option>
                  <option value="fixed">{{ $t('lblConstraintFixed') }}</option>
                </template>
              </select>
            </div>

            <!-- 數值欄位：範圍 -->
            <div v-if="isNumericColumn(constraintSelectedColumn) && constraintDraft.type === 'range'" class="mb-2">
              <div class="row g-2">
                <div class="col-sm-6">
                  <label class="form-label small">{{ $t('lblMin') }}</label>
                  <input type="number" class="form-control form-control-sm" 
                         v-model="constraintDraft.min" :disabled="loading"
                         :placeholder="summary[constraintSelectedColumn] ? String(summary[constraintSelectedColumn].min) : ''">
                </div>
                <div class="col-sm-6">
                  <label class="form-label small">{{ $t('lblMax') }}</label>
                  <input type="number" class="form-control form-control-sm" 
                         v-model="constraintDraft.max" :disabled="loading"
                         :placeholder="summary[constraintSelectedColumn] ? String(summary[constraintSelectedColumn].max) : ''">
                </div>
              </div>
            </div>

            <!-- 數值欄位：僅最小值 -->
            <div v-if="isNumericColumn(constraintSelectedColumn) && constraintDraft.type === 'min'" class="mb-2">
              <label class="form-label small">{{ $t('lblMin') }}</label>
              <input type="number" class="form-control form-control-sm" 
                     v-model="constraintDraft.value" :disabled="loading"
                     :placeholder="summary[constraintSelectedColumn] ? String(summary[constraintSelectedColumn].min) : ''">
            </div>

            <!-- 數值欄位：僅最大值 -->
            <div v-if="isNumericColumn(constraintSelectedColumn) && constraintDraft.type === 'max'" class="mb-2">
              <label class="form-label small">{{ $t('lblMax') }}</label>
              <input type="number" class="form-control form-control-sm" 
                     v-model="constraintDraft.value" :disabled="loading"
                     :placeholder="summary[constraintSelectedColumn] ? String(summary[constraintSelectedColumn].max) : ''">
            </div>

            <!-- 數值欄位：固定值 -->
            <div v-if="isNumericColumn(constraintSelectedColumn) && constraintDraft.type === 'fixed'" class="mb-2">
              <label class="form-label small">{{ $t('lblFixedValue') }}</label>
              <input type="number" class="form-control form-control-sm" 
                     v-model="constraintDraft.value" :disabled="loading">
            </div>

            <!-- 數值欄位：即時範圍警告（超出原始資料範圍時顯示） -->
            <div v-if="constraintNumericWarning" class="alert alert-warning py-2 px-3 small mb-2">
              <i class="fa fa-exclamation-triangle me-1"></i>
              {{ constraintNumericWarning }}
            </div>

            <!-- 類別欄位：限定選項 -->
            <div v-if="!isNumericColumn(constraintSelectedColumn) && constraintDraft.type === 'options'" class="mb-2">
              <label class="form-label small">{{ $t('lblAllowedValues') }}</label>
              <div class="border rounded p-2" style="max-height: 150px; overflow-y: auto;">
                <div v-for="val in getColumnUniqueValues(constraintSelectedColumn)" :key="val" class="form-check">
                  <input class="form-check-input" type="checkbox" 
                         :id="'opt-' + val" :value="val" 
                         v-model="constraintDraft.values" :disabled="loading">
                  <label class="form-check-label small" :for="'opt-' + val">{{ val }}</label>
                </div>
                <div v-if="getColumnUniqueValues(constraintSelectedColumn).length === 0" class="text-muted small">
                  {{ $t('msgNoUniqueValuesDetected') }}
                </div>
              </div>
            </div>

            <!-- 類別欄位：固定值 -->
            <div v-if="!isNumericColumn(constraintSelectedColumn) && constraintDraft.type === 'fixed'" class="mb-2">
              <label class="form-label small">{{ $t('lblFixedValue') }}</label>
              <select v-model="constraintDraft.value" class="form-select form-select-sm" :disabled="loading">
                <option value="">{{ $t('lblPleaseSelect') }}</option>
                <option v-for="val in getColumnUniqueValues(constraintSelectedColumn)" :key="val" :value="val">
                  {{ val }}
                </option>
              </select>
              <div v-if="getColumnUniqueValues(constraintSelectedColumn).length === 0" class="text-muted small mt-1">
                {{ $t('msgNoUniqueValuesDetected') }}
              </div>
            </div>

            <!-- 原始範圍提示 -->
            <div class="form-text small">
              <i class="fa fa-info-circle"></i>
              <template v-if="isNumericColumn(constraintSelectedColumn) && summary[constraintSelectedColumn]">
                {{ $t('lblOriginalRange') }}: {{ summary[constraintSelectedColumn].min }} ~ {{ summary[constraintSelectedColumn].max }}
              </template>
              <template v-else-if="!isNumericColumn(constraintSelectedColumn)">
                {{ $t('lblOriginalValues') }}: {{ getColumnUniqueValues(constraintSelectedColumn).slice(0, 10).join(', ') }}
                <template v-if="getColumnUniqueValues(constraintSelectedColumn).length > 10">...</template>
              </template>
            </div>

            <!-- 按鈕 -->
            <div class="d-flex justify-content-end gap-2 mt-2">
              <button @click="resetConstraintDraft" class="btn btn-outline-secondary btn-sm" type="button" :disabled="loading">
                <i class="fa fa-times me-1"></i> {{ $t('lblCancel') }}
              </button>
              <button @click="addConstraintRule" class="btn btn-outline-primary btn-sm" type="button" :disabled="loading">
                <i class="fa fa-plus me-1"></i> {{ $t('lblAddConstraint') }}
              </button>
            </div>
            <div v-if="errors.constraint" class="text-danger small mt-1">{{ errors.constraint }}</div>
          </div>

          <!-- 已設定的約束列表 -->
          <div v-if="constraintRules.length > 0" class="mb-4">
            <h6 class="small">{{ $t('lblConstraintList') }}</h6>
            <div class="table-responsive card card-body p-2">
              <table class="table table-sm mb-0">
                <thead>
                  <tr>
                    <th class="small">{{ $t('lblColumn') }}</th>
                    <th class="small">{{ $t('lblConstraintType') }}</th>
                    <th class="small">{{ $t('lblConstraintValue') }}</th>
                    <th class="small">{{ $t('lblActions') }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(rule, index) in constraintRules" :key="index">
                    <td class="small">{{ rule.column }}</td>
                    <td class="small">{{ getConstraintTypeLabel(rule.type) }}</td>
                    <td class="small">{{ formatConstraintValue(rule) }}</td>
                    <td>
                      <button class="btn btn-sm btn-outline-danger" type="button" 
                              @click="removeConstraintRule(index)" :disabled="loading">
                        <i class="fa fa-trash"></i>
                      </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- ============ 條件規則 ============ -->
          <hr class="my-3">
          <h6 class="text-muted small mb-2">
            <i class="fa fa-code-branch"></i> {{ $t('lblConditionalRules') }}
          </h6>
          <div class="form-text small mb-2">{{ $t('msgConditionalRulesHint') }}</div>

          <!-- 新增條件規則按鈕 -->
          <div v-if="!showConditionalForm" class="mb-3">
            <button type="button" class="btn btn-outline-primary btn-sm" 
                    @click="openConditionalForm" :disabled="loading">
              <i class="fa fa-plus me-1"></i> {{ $t('lblAddConditionalRule') }}
            </button>
          </div>

          <!-- 條件規則設定卡片 -->
          <div v-if="showConditionalForm" class="card p-3 mb-3 border-primary">
            <!-- IF 子句 -->
            <div class="mb-3">
              <h6 class="small text-primary mb-2">
                <span class="badge bg-primary">{{ $t('lblIf') }}</span>
              </h6>
              <div class="row g-2">
                <div class="col-sm-5">
                  <label class="form-label small">{{ $t('lblWhenColumn') }}</label>
                  <select v-model="conditionalDraft.when.column" 
                          class="form-select form-select-sm" 
                          :disabled="loading"
                          @change="onWhenColumnChange">
                    <option value="">{{ $t('lblPleaseSelect') }}</option>
                    <option v-for="col in preview_data.columns" :key="'wc-' + col" :value="col">
                      {{ col }}
                    </option>
                  </select>
                </div>
                <div class="col-sm-4">
                  <label class="form-label small">{{ $t('lblWhenOperator') }}</label>
                  <select v-model="conditionalDraft.when.operator" 
                          class="form-select form-select-sm" :disabled="loading || !conditionalDraft.when.column">
                    <option v-for="op in availableWhenOperators" :key="op" :value="op">
                      {{ getOperatorLabel(op) }}
                    </option>
                  </select>
                </div>
                <div class="col-sm-3">
                  <label class="form-label small">&nbsp;</label>
                  <span :class="'badge w-100 ' + (conditionalDraft.when.column && isNumericColumn(conditionalDraft.when.column) ? 'bg-secondary' : 'bg-info')" 
                        v-if="conditionalDraft.when.column">
                    {{ isNumericColumn(conditionalDraft.when.column) ? $t('lblNumericColumn') : $t('lblCategoricalColumn') }}
                  </span>
                </div>
              </div>

              <!-- when 值輸入 (依運算子變化) -->
              <div v-if="conditionalDraft.when.column" class="mt-2">
                <!-- 單一值 (==, !=, >=, <=, >, <) -->
                <div v-if="whenNeedsSingleValue">
                  <label class="form-label small">{{ $t('lblWhenValue') }}</label>
                  <!-- 類別欄位：純下拉選擇（只能選現有值） -->
                  <select v-if="!isNumericColumn(conditionalDraft.when.column)"
                          v-model="conditionalDraft.when.value"
                          class="form-select form-select-sm" :disabled="loading">
                    <option value="">{{ $t('lblPleaseSelect') }}</option>
                    <option v-for="val in getColumnUniqueValues(conditionalDraft.when.column)"
                            :key="'wv-' + val" :value="val">{{ val }}</option>
                  </select>
                  <!-- 數值欄位 -->
                  <input v-else type="number"
                         class="form-control form-control-sm"
                         v-model="conditionalDraft.when.value" :disabled="loading">
                </div>

                <!-- 多值列表 (in, not_in) -->
                <div v-if="whenNeedsValuesList">
                  <label class="form-label small">{{ $t('lblWhenValues') }}</label>
                  <div class="border rounded p-2" style="max-height: 120px; overflow-y: auto;">
                    <div v-for="val in getColumnUniqueValues(conditionalDraft.when.column)" 
                         :key="'wvl-' + val" class="form-check form-check-inline">
                      <input class="form-check-input" type="checkbox" 
                             :id="'wopt-' + val" :value="val" 
                             v-model="conditionalDraft.when.values" :disabled="loading">
                      <label class="form-check-label small" :for="'wopt-' + val">{{ val }}</label>
                    </div>
                    <div v-if="getColumnUniqueValues(conditionalDraft.when.column).length === 0" class="text-muted small">
                      {{ $t('msgNoUniqueValuesDetected') }}
                    </div>
                  </div>
                </div>

                <!-- between 範圍 -->
                <div v-if="conditionalDraft.when.operator === 'between'">
                  <div class="row g-2">
                    <div class="col-6">
                      <label class="form-label small">{{ $t('lblMin') }}</label>
                      <input type="number" class="form-control form-control-sm" 
                             v-model="conditionalDraft.when.min" :disabled="loading">
                    </div>
                    <div class="col-6">
                      <label class="form-label small">{{ $t('lblMax') }}</label>
                      <input type="number" class="form-control form-control-sm" 
                             v-model="conditionalDraft.when.max" :disabled="loading">
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- THEN 子句 -->
            <div class="mb-3">
              <h6 class="small text-success mb-2">
                <span class="badge bg-success">{{ $t('lblThen') }}</span>
              </h6>
              <div class="row g-2">
                <div class="col-sm-6">
                  <label class="form-label small">{{ $t('lblThenColumn') }}</label>
                  <select v-model="conditionalDraft.then.column" 
                          class="form-select form-select-sm" 
                          :disabled="loading"
                          @change="onThenColumnChange">
                    <option value="">{{ $t('lblPleaseSelect') }}</option>
                    <option v-for="col in preview_data.columns" :key="'tc-' + col" :value="col">
                      {{ col }}
                    </option>
                  </select>
                </div>
                <div class="col-sm-6">
                  <label class="form-label small">{{ $t('lblConstraintType') }}</label>
                  <select v-model="conditionalDraft.then.type" 
                          class="form-select form-select-sm" :disabled="loading || !conditionalDraft.then.column">
                    <template v-if="conditionalDraft.then.column && isNumericColumn(conditionalDraft.then.column)">
                      <option value="range">{{ $t('lblConstraintRange') }}</option>
                      <option value="min">{{ $t('lblConstraintMin') }}</option>
                      <option value="max">{{ $t('lblConstraintMax') }}</option>
                      <option value="fixed">{{ $t('lblConstraintFixed') }}</option>
                    </template>
                    <template v-else-if="conditionalDraft.then.column">
                      <option value="options">{{ $t('lblConstraintOptions') }}</option>
                      <option value="fixed">{{ $t('lblConstraintFixed') }}</option>
                    </template>
                  </select>
                </div>
              </div>

              <!-- then 值輸入 (依類型變化) -->
              <div v-if="conditionalDraft.then.column" class="mt-2">
                <!-- 數值範圍 -->
                <div v-if="isNumericColumn(conditionalDraft.then.column) && conditionalDraft.then.type === 'range'">
                  <div class="row g-2">
                    <div class="col-6">
                      <label class="form-label small">{{ $t('lblMin') }}</label>
                      <input type="number" class="form-control form-control-sm" 
                             v-model="conditionalDraft.then.min" :disabled="loading">
                    </div>
                    <div class="col-6">
                      <label class="form-label small">{{ $t('lblMax') }}</label>
                      <input type="number" class="form-control form-control-sm" 
                             v-model="conditionalDraft.then.max" :disabled="loading">
                    </div>
                  </div>
                </div>

                <!-- 數值最小/最大值 -->
                <div v-if="isNumericColumn(conditionalDraft.then.column) && (conditionalDraft.then.type === 'min' || conditionalDraft.then.type === 'max')">
                  <label class="form-label small">{{ conditionalDraft.then.type === 'min' ? $t('lblMin') : $t('lblMax') }}</label>
                  <input type="number" class="form-control form-control-sm" 
                         v-model="conditionalDraft.then.value" :disabled="loading">
                </div>

                <!-- 固定值 -->
                <div v-if="conditionalDraft.then.type === 'fixed'">
                  <label class="form-label small">{{ $t('lblFixedValue') }}</label>
                  <!-- 類別欄位：純下拉選擇 -->
                  <select v-if="!isNumericColumn(conditionalDraft.then.column)"
                          v-model="conditionalDraft.then.value"
                          class="form-select form-select-sm" :disabled="loading">
                    <option value="">{{ $t('lblPleaseSelect') }}</option>
                    <option v-for="val in getColumnUniqueValues(conditionalDraft.then.column)"
                            :key="'tv-' + val" :value="val">{{ val }}</option>
                  </select>
                  <!-- 數值欄位 -->
                  <input v-else type="number"
                         class="form-control form-control-sm" 
                         v-model="conditionalDraft.then.value" :disabled="loading">
                </div>

                <!-- 類別限定選項 -->
                <div v-if="!isNumericColumn(conditionalDraft.then.column) && conditionalDraft.then.type === 'options'">
                  <label class="form-label small">{{ $t('lblAllowedValues') }}</label>
                  <div class="border rounded p-2" style="max-height: 120px; overflow-y: auto;">
                    <div v-for="val in getColumnUniqueValues(conditionalDraft.then.column)" 
                         :key="'tvo-' + val" class="form-check form-check-inline">
                      <input class="form-check-input" type="checkbox" 
                             :id="'topt-' + val" :value="val" 
                             v-model="conditionalDraft.then.values" :disabled="loading">
                      <label class="form-check-label small" :for="'topt-' + val">{{ val }}</label>
                    </div>
                    <div v-if="getColumnUniqueValues(conditionalDraft.then.column).length === 0" class="text-muted small">
                      {{ $t('msgNoUniqueValuesDetected') }}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- 按鈕 -->
            <div class="d-flex justify-content-end gap-2">
              <button @click="resetConditionalDraft" class="btn btn-outline-secondary btn-sm" type="button" :disabled="loading">
                <i class="fa fa-times me-1"></i> {{ $t('lblCancel') }}
              </button>
              <button @click="addConditionalRule" class="btn btn-outline-primary btn-sm" type="button" :disabled="loading">
                <i class="fa fa-plus me-1"></i> {{ $t('lblAddConditionalRule') }}
              </button>
            </div>
            <div v-if="errors.conditional" class="text-danger small mt-1">{{ errors.conditional }}</div>
          </div>

          <!-- 已設定的條件規則列表 -->
          <div v-if="conditionalRules.length > 0">
            <h6 class="small">{{ $t('lblConditionalList') }}</h6>
            <div class="table-responsive card card-body p-2">
              <table class="table table-sm mb-0">
                <thead>
                  <tr>
                    <th class="small">{{ $t('lblIf') }}</th>
                    <th class="small">{{ $t('lblThen') }}</th>
                    <th class="small">{{ $t('lblActions') }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(rule, index) in conditionalRules" :key="'cr-' + index">
                    <td class="small">{{ formatWhenClause(rule.when) }}</td>
                    <td class="small">{{ formatThenClause(rule.then) }}</td>
                    <td>
                      <button class="btn btn-sm btn-outline-danger" type="button" 
                              @click="removeConditionalRule(index)" :disabled="loading">
                        <i class="fa fa-trash"></i>
                      </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 進階參數(可收合) -->
    <div class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblAdvancedParams') }}</label>
      <div class="col-sm-8">
        <button type="button" class="btn btn-outline-secondary btn-sm" @click="showAdvanced = !showAdvanced">
          <i :class="showAdvanced ? 'fa fa-chevron-up' : 'fa fa-chevron-down'"></i>
          {{ showAdvanced ? $t('lblHide') : $t('lblShow') }}
        </button>
      </div>
    </div>

    <!-- CTGAN 參數 -->
    <template v-if="showAdvanced && selected.generator_type === 'ctgan'">
      <!-- 訓練參數 -->
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label text-muted small">{{ $t('lblTrainingParams') }}</label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.ctgan.epochs" type="text" class="form-control" 
                 id="floatingEpochs" :disabled="loading">
          <label for="floatingEpochs" style="margin-left:9px;">epochs</label>
          <div v-if="errors.epochs" class="text-danger small">{{ errors.epochs }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.ctgan.batch_size" type="text" class="form-control" 
                 id="floatingBatchSize" :disabled="loading">
          <label for="floatingBatchSize" style="margin-left:9px;">batch_size</label>
          <div v-if="errors.batch_size" class="text-danger small">{{ errors.batch_size }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.ctgan.discriminator_steps" type="text" class="form-control" 
                 id="floatingDiscSteps" :disabled="loading">
          <label for="floatingDiscSteps" style="margin-left:9px;">disc_steps</label>
          <div v-if="errors.discriminator_steps" class="text-danger small">{{ errors.discriminator_steps }}</div>
        </div>
      </div>

      <!-- Learning Rate(論文新增) -->
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label text-muted small">{{ $t('lblLearningRateParams') }}</label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.ctgan.generator_lr" type="text" class="form-control" 
                 id="floatingGenLR" :disabled="loading">
          <label for="floatingGenLR" style="margin-left:9px;">gen_lr</label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.ctgan.discriminator_lr" type="text" class="form-control" 
                 id="floatingDiscLR" :disabled="loading">
          <label for="floatingDiscLR" style="margin-left:9px;">disc_lr</label>
        </div>
        <div class="col-sm-4">
          <div class="form-text">{{ $t('msgLearningRateHint') }}</div>
        </div>
      </div>
      
      <!-- 網路架構參數 -->
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label text-muted small">{{ $t('lblNetworkParams') }}</label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.ctgan.generator_dim" type="text" class="form-control" 
                 id="floatingGenDim" :disabled="loading">
          <label for="floatingGenDim" style="margin-left:9px;">gen_dim</label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.ctgan.discriminator_dim" type="text" class="form-control" 
                 id="floatingDiscDim" :disabled="loading">
          <label for="floatingDiscDim" style="margin-left:9px;">disc_dim</label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.ctgan.embedding_dim" type="text" class="form-control" 
                 id="floatingEmbDim" :disabled="loading">
          <label for="floatingEmbDim" style="margin-left:9px;">embed_dim</label>
        </div>
      </div>

      <!-- 參數說明 (NeurIPS 2019 論文依據) -->
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label"></label>
        <div class="col-sm-8">
          <div class="alert alert-light small mb-0">
            <p class="mb-1"><strong>epochs</strong>: {{ $t('msgParamEpochsDesc') }}</p>
            <p class="mb-1"><strong>batch_size</strong>: {{ $t('msgParamBatchSizeDesc') }}</p>
            <p class="mb-1"><strong>disc_steps</strong>: {{ $t('msgParamDiscStepsDesc') }}</p>
            <p class="mb-1"><strong>gen_lr / disc_lr</strong>: {{ $t('msgParamLrDesc') }}</p>
            <p class="mb-1"><strong>gen_dim / disc_dim / embed_dim</strong>: {{ $t('msgParamDimDesc') }}</p>
            <p class="mb-0 text-info"><i class="fa fa-book me-1"></i>{{ $t('msgParamCtganPaperRef') || '參數依據: CTGAN (NeurIPS 2019) + SDV 官方最佳實踐' }}</p>
          </div>
        </div>
      </div>
    </template>

    <!-- TVAE 參數 -->
    <template v-if="showAdvanced && selected.generator_type === 'tvae'">
      <!-- 訓練參數（對應 CTGAN 的訓練參數行） -->
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label text-muted small">{{ $t('lblTrainingParams') }}</label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tvae.epochs" type="text" class="form-control" 
                 id="floatingTvaeEpochs" :disabled="loading">
          <label for="floatingTvaeEpochs" style="margin-left:9px;">epochs</label>
          <div v-if="errors.tvae_epochs" class="text-danger small">{{ errors.tvae_epochs }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tvae.batch_size" type="text" class="form-control" 
                 id="floatingTvaeBatchSize" :disabled="loading">
          <label for="floatingTvaeBatchSize" style="margin-left:9px;">batch_size</label>
          <div v-if="errors.tvae_batch_size" class="text-danger small">{{ errors.tvae_batch_size }}</div>
        </div>
      </div>

      <!-- 正則化參數（對應 CTGAN 的學習率參數行） -->
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label text-muted small">{{ $t('lblRegularizationParams') }}</label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tvae.l2scale" type="text" class="form-control" 
                 id="floatingL2Scale" :disabled="loading">
          <label for="floatingL2Scale" style="margin-left:9px;">l2scale</label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tvae.loss_factor" type="text" class="form-control" 
                 id="floatingLossFactor" :disabled="loading">
          <label for="floatingLossFactor" style="margin-left:9px;">loss_factor</label>
        </div>
        <div class="col-sm-4">
          <div class="form-text">{{ $t('msgRegularizationHint') }}</div>
        </div>
      </div>
      
      <!-- 網路架構參數（對應 CTGAN 的網路架構行） -->
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label text-muted small">{{ $t('lblNetworkParams') }}</label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tvae.compress_dims" type="text" class="form-control" 
                 id="floatingCompressDim" :disabled="loading">
          <label for="floatingCompressDim" style="margin-left:9px;">compress</label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tvae.decompress_dims" type="text" class="form-control" 
                 id="floatingDecompressDim" :disabled="loading">
          <label for="floatingDecompressDim" style="margin-left:9px;">decompress</label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tvae.embedding_dim" type="text" class="form-control" 
                 id="floatingTvaeEmbDim" :disabled="loading">
          <label for="floatingTvaeEmbDim" style="margin-left:9px;">embed_dim</label>
        </div>
      </div>

      <!-- 參數說明（NeurIPS 2019 論文依據） -->
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label"></label>
        <div class="col-sm-8">
          <div class="alert alert-light small mb-0">
            <p class="mb-1"><strong>epochs</strong>: {{ $t('msgParamTvaeEpochsDesc') }}</p>
            <p class="mb-1"><strong>batch_size</strong>: {{ $t('msgParamBatchSizeDesc') }}</p>
            <p class="mb-1"><strong>l2scale / loss_factor</strong>: {{ $t('msgParamRegularizationDesc') }}</p>
            <p class="mb-1"><strong>compress / decompress / embed_dim</strong>: {{ $t('msgParamTvaeDimDesc') }}</p>
            <p class="mb-0 text-info"><i class="fa fa-book me-1"></i>{{ $t('msgParamTvaePaperRef') || '參數依據: TVAE (NeurIPS 2019) + SDV 官方最佳實踐' }}</p>
          </div>
        </div>
      </div>
    </template>

    <!-- TabDDPM 進階參數 -->
    <template v-if="showAdvanced && selected.generator_type === 'tabddpm'">
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label">{{ $t('lblDiffusionParams') || 'Diffusion Params' }}</label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tabddpm.n_iter" type="text" class="form-control" 
                 id="floatingDdpmNiter" :disabled="loading">
          <label for="floatingDdpmNiter" style="margin-left:9px;">n_iter</label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tabddpm.batch_size" type="text" class="form-control" 
                 id="floatingDdpmBatch" :disabled="loading">
          <label for="floatingDdpmBatch" style="margin-left:9px;">batch_size</label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tabddpm.num_timesteps" type="text" class="form-control" 
                 id="floatingDdpmTimesteps" :disabled="loading">
          <label for="floatingDdpmTimesteps" style="margin-left:9px;">timesteps</label>
        </div>
      </div>

      <!-- TabDDPM 即時偏離警告 (僅在手動模式顯示) -->
      <div v-if="!selected.use_auto_params && tabddpmDeviationWarnings.length > 0" class="row mb-2">
        <label class="col-sm-3 col-form-label"></label>
        <div class="col-sm-8">
          <div class="alert alert-warning py-2 px-3 small mb-0">
            <i class="fa fa-exclamation-triangle me-1"></i>
            <strong>{{ $t('msgDdpmDeviationTitle') || '偏離論文建議值' }}</strong>
            <ul class="mb-1 mt-1" style="padding-left: 1.2rem;">
              <li v-for="w in tabddpmDeviationWarnings" :key="w">{{ w }}</li>
            </ul>
            <div class="text-muted small">
              {{ $t('msgDdpmDeviationNote') || '後端會尊重您的輸入值，但品質可能低於論文基準。建議：開啟自動參數或使用論文建議值。' }}
            </div>
          </div>
        </div>
      </div>

      <!-- TabDDPM 參數說明 (ICML 2023 論文依據) -->
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label"></label>
        <div class="col-sm-8">
          <div class="alert alert-light small mb-0">
            <p class="mb-1"><strong>n_iter</strong>: {{ $t('msgParamDdpmNiterDesc') || '訓練迭代次數，論文建議 5000/10000/20000' }}</p>
            <p class="mb-1"><strong>batch_size</strong>: {{ $t('msgParamDdpmBatchDesc') || '批次大小，論文建議 256 或 4096' }}</p>
            <p class="mb-1"><strong>timesteps</strong>: {{ $t('msgParamDdpmTimestepsDesc') || '擴散步數，100=快速 / 1000=高品質' }}</p>
            <p class="mb-0 text-info"><i class="fa fa-book me-1"></i>{{ $t('msgParamDdpmPaperRef') || '參數依據: TabDDPM (ICML 2023) Table 1' }}</p>
          </div>
        </div>
      </div>
    </template>

    <!-- 提交按鈕 -->
    <div class="col-12">
      <!-- 訓練提示 - 根據資料規模和模型給出實用建議 -->
      <div v-if="!loading" class="alert py-2 mb-3" :class="{
        'alert-success': selected.generator_type === 'tvae',
        'alert-warning': selected.generator_type === 'ctgan',
        'alert-danger': selected.generator_type === 'tabddpm'
      }">
        <div class="d-flex align-items-start">
          <i class="fa fa-info-circle me-2 mt-1"></i>
          <div class="flex-grow-1">
            <!-- 資料規模分析 -->
            <div class="mb-2">
              <strong>{{ $t('lblDataAnalysis') || '資料分析' }}:</strong>
              <div class="row mt-1 small">
                <div class="col-auto">
                  <span class="text-muted">{{ $t('lblSampleCount') || '樣本數' }}:</span>
                  <span class="badge bg-secondary ms-1">{{ (preview_data.total_rows || 0).toLocaleString() }}</span>
                  <span v-if="rowWarningLevel" :class="'text-' + rowWarningLevel.level" class="ms-1">
                    <i class="fa fa-exclamation-circle"></i>
                  </span>
                </div>
                <div class="col-auto">
                  <span class="text-muted">{{ $t('lblColumnCount') || '欄位數' }}:</span>
                  <span class="badge ms-1" :class="preview_data.total_columns > 50 ? 'bg-warning text-dark' : 'bg-secondary'">
                    {{ preview_data.total_columns || 0 }}
                  </span>
                  <span v-if="columnWarningLevel" :class="'text-' + columnWarningLevel.level" class="ms-1">
                    <i class="fa fa-exclamation-circle"></i>
                  </span>
                </div>
                <div class="col-auto">
                  <span class="text-muted">{{ $t('lblOverallSize') || '整體規模' }}:</span>
                  <span class="badge ms-1" :class="dataSizeLevel.class">{{ dataSizeLevel.label }}</span>
                </div>
              </div>
              
              <!-- 欄位數警告 -->
              <div v-if="columnWarningLevel && columnWarningLevel.level !== 'info'" class="small mt-1" :class="'text-' + columnWarningLevel.level">
                <i class="fa fa-warning me-1"></i>{{ columnWarningLevel.msg }}
              </div>
              
              <!-- 樣本數警告 -->
              <div v-if="rowWarningLevel && rowWarningLevel.level !== 'info'" class="small mt-1" :class="'text-' + rowWarningLevel.level">
                <i class="fa fa-warning me-1"></i>{{ rowWarningLevel.msg }}
              </div>
            </div>
            
            <!-- 模型速度提示 -->
            <div class="mb-2">
              <strong>{{ $t('lblModelSpeed') || '模型速度' }}:</strong>
              <span v-if="selected.generator_type === 'tvae'" class="text-success">
                <i class="fa fa-bolt me-1"></i>{{ $t('lblFastModel') || '較快（VAE 架構）' }}
              </span>
              <span v-else-if="selected.generator_type === 'ctgan'" class="text-warning">
                <i class="fa fa-clock-o me-1"></i>{{ $t('lblMediumModel') || '中等（GAN 架構）' }}
              </span>
              <span v-else-if="selected.generator_type === 'tabddpm'" class="text-danger fw-bold">
                <i class="fa fa-hourglass-half me-1"></i>{{ $t('lblSlowModel') || '非常慢（擴散模型）' }}
              </span>
            </div>
            
            <!-- TabDDPM 特別警告 -->
            <div v-if="selected.generator_type === 'tabddpm'" class="alert alert-danger py-2 px-3 mb-2">
              <i class="fa fa-exclamation-triangle me-1"></i>
              <strong>TabDDPM 警告：</strong>
              <ul class="mb-0 mt-1 small">
                <li>{{ $t('msgDdpmWarn1') || '擴散模型訓練時間遠超 CTGAN/TVAE（可能 5-20 倍）' }}</li>
                <li v-if="preview_data.total_columns > 50">
                  <strong class="text-warning">⚠️ 您的資料有 {{ preview_data.total_columns }} 欄，訓練可能需要數小時！</strong>
                </li>
                <li v-if="preview_data.total_rows > 10000">
                  <strong>{{ $t('msgDdpmWarnHighRow') || '樣本數超過 1 萬筆，訓練時間會顯著增加' }}</strong>
                </li>
                <li v-if="preview_data.total_rows > 50000">
                  <strong class="text-warning">⚠️ 樣本數超過 5 萬筆，建議考慮抽樣或使用 CTGAN</strong>
                </li>
                <li><strong>{{ $t('msgDdpmWarn4') || '強烈建議：先用 CTGAN 測試，確認需求後再用 TabDDPM' }}</strong></li>
              </ul>
            </div>
            
            <!-- 高維度資料警告 -->
            <div v-if="preview_data.total_columns > 50 && selected.generator_type !== 'tabddpm'" class="alert alert-warning py-1 px-2 mb-2 small">
              <i class="fa fa-warning me-1"></i>
              {{ $t('msgHighDimWarning') || '您的資料有較多欄位，訓練時間會相應增加。' }}
            </div>
            
            <!-- 通用提醒 -->
            <div class="small border-top pt-2 mt-2">
              <i class="fa fa-lightbulb-o me-1 text-warning"></i>
              <strong>{{ $t('lblTips') || '提示' }}:</strong>
              <ul class="mb-0 mt-1">
                <li>{{ $t('msgTip1') || '訓練期間請勿關閉或重新整理頁面' }}</li>
                <li>{{ $t('msgTip2') || '欄位數對訓練時間影響最大（尤其是 TabDDPM）' }}</li>
                <li v-if="preview_data.total_columns > 80">
                  <strong class="text-danger">{{ $t('msgTipReduceCols') || '建議進行特徵篩選，移除不重要的欄位以加速訓練' }}</strong>
                </li>
                <li v-if="preview_data.total_rows > 100000">
                  <strong class="text-warning">{{ $t('msgTipSample') || '建議對資料進行抽樣（如隨機抽取 10-20%），或分批生成' }}</strong>
                </li>
                <li v-if="selected.generator_type === 'tabddpm'">{{ $t('msgTip3') || '若訓練時間過長，可嘗試減少 n_iter 參數或改用 CTGAN' }}</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 訓練中狀態 -->
      <div v-if="loading" class="alert alert-danger py-3 mb-3">
        <div class="d-flex align-items-center mb-2">
          <span class="spinner-border spinner-border-sm me-2"></span>
          <strong class="fs-5">{{ $t('lblTrainingInProgress') || '訓練進行中...' }}</strong>
          <span class="badge ms-2" :class="{
            'bg-success': selected.generator_type === 'tvae',
            'bg-warning text-dark': selected.generator_type === 'ctgan',
            'bg-danger': selected.generator_type === 'tabddpm'
          }">{{ selected.generator_type.toUpperCase() }}</span>
        </div>
        
        <div class="progress mb-2" style="height: 6px;">
          <div class="progress-bar progress-bar-striped progress-bar-animated bg-danger" style="width: 100%"></div>
        </div>
        
        <div class="d-flex align-items-center text-danger">
          <i class="fa fa-exclamation-triangle me-2"></i>
          <strong>{{ $t('msgTrainingWarning') || '請勿關閉或重新整理此頁面！' }}</strong>
        </div>
        
        <div v-if="selected.generator_type === 'tabddpm'" class="mt-2 small text-muted">
          <i class="fa fa-coffee me-1"></i>
          {{ $t('msgTabddpmTip') || 'TabDDPM 訓練需要較長時間，請耐心等待...' }}
        </div>
      </div>
      
      <button v-if="!loading" type="submit" class="btn btn-primary btn-lg">
        <i class="fa fa-cogs me-1"></i>{{ $t('lblGenerate') }}
      </button>
      <button v-if="loading" class="btn btn-danger btn-lg" type="button" disabled>
        <span class="spinner-border spinner-border-sm me-1" aria-hidden="true"></span>
        {{ $t('lblGenerating') || '生成中' }}...
      </button>
    </div>
  </form>

  <!-- 結果區域 -->
  <div v-if="output" class="results-section">
    <hr class="my-4">
    
    <!-- 基本資訊 -->
    <div class="card mb-4">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h5 class="mb-0">{{ $t('lblGenerationResult') }}</h5>
        <div>
          <a :href="downloadUrl(output.synthetic_file || 'synthetic_data.csv')" class="btn btn-success btn-sm me-2">
            <i class="fa fa-download me-1"></i>{{ $t('lblDownloadData') }}
          </a>
          <a :href="downloadUrl('quality_report.json')" class="btn btn-outline-secondary btn-sm">
            <i class="fa fa-file-text me-1"></i>{{ $t('lblDownloadReport') }}
          </a>
        </div>
      </div>
      <div class="card-body">
        <div class="row">
          <div class="col-md-3">
            <strong>{{ $t('lblGenerator') }}:</strong> {{ output.generator_type.toUpperCase() }}
            <span v-if="output.train_mode === 'grouped'" class="badge bg-success ms-1">{{ $t('lblTrainModeGrouped') }}</span>
          </div>
          <div class="col-md-3">
            <strong>{{ $t('lblOriginalRows') }}:</strong> {{ output.original_rows }}
          </div>
          <div class="col-md-3">
            <strong>{{ $t('lblSyntheticRows') }}:</strong> {{ output.synthetic_rows }}
            <span v-if="output.tgant && output.tgant.summary" class="text-muted small">({{ $t('lblFiltered') }})</span>
          </div>
          <div class="col-md-3">
            <strong>{{ $t('lblTrainingTime') }}:</strong> {{ formatTrainingTime(output.training_time_seconds) }}
          </div>
        </div>
        <!-- TGANT 篩選結果 -->
        <div v-if="output.tgant && output.tgant.summary" class="row mt-3">
          <div class="col-12">
            <div class="alert alert-info mb-0">
              <strong><i class="fa fa-filter me-1"></i>{{ $t('lblTGANTFilterResult') }}</strong>
              <div class="row mt-2">
                <div class="col-md-3">
                  <small>{{ $t('lblBeforeDistillation') }}:</small> 
                  <strong>{{ output.tgant.summary.original_synthetic_rows || '-' }}</strong>
                </div>
                <div class="col-md-3">
                  <small>{{ $t('lblAfterDistillation') }}:</small> 
                  <strong>{{ output.tgant.summary.final_synthetic_rows || '-' }}</strong>
                </div>
                <div class="col-md-3">
                  <small>Recall (TRTS/TSTR):</small>
                  <strong>
                    {{ formatPercent(output.tgant.summary.final_recall_trts) }} / 
                    {{ formatPercent(output.tgant.summary.final_recall_tstr) }}
                  </strong>
                </div>
                <div class="col-md-3">
                  <small>{{ $t('lblTargetAchieved') }} (≥{{ output.tgant.summary.threshold ? (output.tgant.summary.threshold * 100).toFixed(0) : '95' }}%):</small>
                  <span :class="output.tgant.summary.converged ? 'badge bg-success' : 'badge bg-warning'">
                    {{ output.tgant.summary.converged ? $t('lblConverged') : $t('lblNotConverged') }}
                  </span>
                  <small v-if="output.tgant.summary.metric" class="ms-1 text-muted">
                    ({{ output.tgant.summary.metric }})
                  </small>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Fidelity 評估 -->
    <div class="card mb-4">
      <div class="card-header">
        <h6 class="mb-0">
          <i class="fa fa-check-circle me-2"></i>Fidelity({{ $t('lblStatisticalSimilarity') }})
        </h6>
      </div>
      <div class="card-body">
        <div class="row">
          <div class="col-md-6">
            <p><strong>Single Column Similarity:</strong> 
              <span :class="getScoreClass(output.evaluation.fidelity.single_column_similarity)">
                {{ output.evaluation.fidelity.single_column_similarity != null ? (output.evaluation.fidelity.single_column_similarity * 100).toFixed(2) + '%' : '-' }}
              </span>
            </p>
            <p><strong>Multi Column Similarity:</strong> 
              <span :class="getScoreClass(output.evaluation.fidelity.multi_column_similarity)">
                {{ output.evaluation.fidelity.multi_column_similarity != null ? (output.evaluation.fidelity.multi_column_similarity * 100).toFixed(2) + '%' : '-' }}
              </span>
            </p>
          </div>
          <div class="col-md-6">
            <p><strong>KL Divergence (avg):</strong> {{ output.evaluation.fidelity.kl_divergence != null ? output.evaluation.fidelity.kl_divergence.toFixed(4) : '-' }}</p>
            <p><strong>JS Divergence (avg):</strong> {{ output.evaluation.fidelity.js_divergence != null ? output.evaluation.fidelity.js_divergence.toFixed(4) : '-' }}</p>
          </div>
        </div>
        <p class="text-muted small mb-0">
          <i class="fa fa-info-circle me-1"></i>
          {{ $t('msgFidelityExplain') }}
        </p>
      </div>
    </div>

    <!-- Privacy 評估 -->
    <div class="card mb-4">
      <div class="card-header">
        <h6 class="mb-0">
          <i class="fa fa-shield me-2"></i>Privacy({{ $t('lblPrivacyProtection') }})
        </h6>
      </div>
      <div class="card-body">
        <p><strong>DCR Baseline Protection:</strong> 
          <span :class="getScoreClass(output.evaluation.privacy.score)">
            {{ output.evaluation.privacy.score != null ? (output.evaluation.privacy.score * 100).toFixed(2) + '%' : '-' }}
          </span>
        </p>
        <p><strong>Median DCR (Synthetic):</strong> {{ output.evaluation.privacy.median_dcr_synthetic != null ? output.evaluation.privacy.median_dcr_synthetic.toFixed(4) : '-' }}</p>
        <p><strong>Median DCR (Baseline):</strong> {{ output.evaluation.privacy.median_dcr_baseline != null ? output.evaluation.privacy.median_dcr_baseline.toFixed(4) : '-' }}</p>
        <p class="text-muted small mb-0">
          <i class="fa fa-info-circle me-1"></i>
          {{ $t('msgPrivacyExplain') }}
        </p>
      </div>
    </div>

    <!-- Utility 評估 -->
    <div class="card mb-4">
      <div class="card-header">
        <h6 class="mb-0">
          <i class="fa fa-line-chart me-2"></i>Utility({{ $t('lblPracticalUtility') }})
        </h6>
      </div>
      <div class="card-body">
        <!-- Discriminator Measure -->
        <h6>Discriminator Measure</h6>
        <p><strong>Random Forest Accuracy:</strong> 
          <span :class="getDiscriminatorClass(output.evaluation.utility.discriminator_measure?.random_forest?.accuracy)">
            {{ getDiscriminatorAccuracy('random_forest') }}%
          </span>
          <span class="text-muted small">({{ $t('lblCloserTo50Better') }})</span>
        </p>
        <p><strong>XGBoost Accuracy:</strong> 
          <span :class="getDiscriminatorClass(output.evaluation.utility.discriminator_measure?.xgboost?.accuracy)">
            {{ getDiscriminatorAccuracy('xgboost') }}%
          </span>
          <span class="text-muted small">({{ $t('lblCloserTo50Better') }})</span>
        </p>
        <p class="text-muted small">
          <i class="fa fa-info-circle me-1"></i>
          {{ $t('msgDiscriminatorExplain') }}
        </p>

        <!-- ML Efficacy (TSTR) -->
        <template v-if="output.evaluation.utility.ml_efficacy && output.evaluation.utility.ml_efficacy.baseline">
          <hr>
          <h6>ML Efficacy (TSTR)</h6>
          
          <!-- 二分類表格 -->
          <div v-if="output.evaluation.utility.ml_efficacy.is_binary" class="table-responsive">
            <table class="table table-sm table-bordered">
              <thead class="table-light">
                <tr>
                  <th></th>
                  <th>Accuracy</th>
                  <th>Recall</th>
                  <th>Specificity</th>
                  <th>Precision</th>
                  <th>NPV</th>
                  <th>F1</th>
                  <th>F2</th>
                  <th>G-mean</th>
                  <th>AUC</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Baseline</strong></td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.accuracy) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.recall) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.specificity) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.precision) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.npv) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.f1) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.f2) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.gmean) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.auc) }}</td>
                </tr>
                <tr>
                  <td><strong>TSTR</strong></td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.accuracy) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.recall) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.specificity) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.precision) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.npv) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.f1) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.f2) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.gmean) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.auc) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          
          <!-- 多分類表格 -->
          <div v-else class="table-responsive">
            <table class="table table-sm table-bordered">
              <thead class="table-light">
                <tr>
                  <th></th>
                  <th>Accuracy</th>
                  <th>Recall (Macro)</th>
                  <th>Precision (Macro)</th>
                  <th>F1 (Macro)</th>
                  <th>G-mean</th>
                  <th>AUC (Macro)</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Baseline</strong></td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.accuracy) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.recall_macro) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.precision_macro) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.f1_macro) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.gmean) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.baseline.auc) }}</td>
                </tr>
                <tr v-if="output.evaluation.utility.ml_efficacy.tstr && !output.evaluation.utility.ml_efficacy.tstr.error">
                  <td><strong>TSTR</strong></td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.accuracy) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.recall_macro || output.evaluation.utility.ml_efficacy.tstr.recall) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.precision_macro || output.evaluation.utility.ml_efficacy.tstr.precision) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.f1_macro || output.evaluation.utility.ml_efficacy.tstr.f1) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.gmean) }}</td>
                  <td>{{ formatPercent(output.evaluation.utility.ml_efficacy.tstr.auc) }}</td>
                </tr>
              </tbody>
            </table>
            
            <!-- Per-class Recall (多分類) -->
            <div v-if="output.evaluation.utility.ml_efficacy.baseline.per_class_recall" class="mt-2">
              <small class="text-muted"><strong>Per-class Recall (Baseline):</strong></small>
              <div class="d-flex flex-wrap gap-2 mt-1">
                <span v-for="(recall, className) in output.evaluation.utility.ml_efficacy.baseline.per_class_recall" 
                      :key="className" 
                      class="badge bg-secondary">
                  {{ className }}: {{ formatPercent(recall) }}
                </span>
              </div>
            </div>
          </div>
          
          <p class="text-muted small mb-0">
            <i class="fa fa-info-circle me-1"></i>
            {{ $t('msgMLEfficacyExplain') }}
          </p>
        </template>
      </div>
    </div>

    <!-- 合成資料預覽 -->
    <div class="card mb-4" v-if="syntheticPreviewData.rows.length > 0">
      <div class="card-header">
        <h6 class="mb-0">{{ $t('lblSyntheticPreview') }}</h6>
      </div>
      <div class="card-body">
        <div class="table-responsive" style="max-height: 300px; overflow-y: auto;">
          <table class="table table-sm table-bordered table-striped">
            <thead class="table-light sticky-top">
              <tr>
                <th v-for="col in syntheticPreviewData.columns" :key="col">{{ col }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, idx) in syntheticPreviewData.rows" :key="idx">
                <td v-for="col in syntheticPreviewData.columns" :key="col">{{ row[col] }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p class="text-muted small mt-2 mb-0">
          {{ $t('msgShowingFirst10Rows') }}
        </p>
      </div>
    </div>
  </div>

  <!-- Modal 通知 -->
  <ModalNotification
    ref="modalNotificationRef"
    :title="modal.title"
    :content="modal.content"
    :icon="modal.icon"
  />
</template>

<script>
import axios from 'axios'
import { Collapse } from 'bootstrap'
import ModalNotification from '@/components/ModalNotification.vue'

export default {
  name: 'SyntheticView',
  components: {
    ModalNotification
  },
  data() {
    return {
      loading: false,
      showAdvanced: false,
      showConstraints: false,
      constraintSelectedColumn: '',
      constraintDraft: {
        type: 'range',
        min: '',
        max: '',
        value: '',
        values: []
      },
      constraintRules: [],
      // 條件規則
      showConditionalForm: false,
      conditionalDraft: {
        when: {
          column: '',
          operator: '==',
          value: '',
          values: [],
          min: '',
          max: ''
        },
        then: {
          column: '',
          type: 'range',
          min: '',
          max: '',
          value: '',
          values: []
        }
      },
      conditionalRules: [],
      files: [],
      selected: {
        generator_type: 'ctgan',
        data: '',
        label_column: 'none',  // 單標籤（僅能選擇一個欄位）
        num_samples_type: '1x',
        custom_num_samples: '',
        train_mode: 'mixed',    // 預設混合訓練（因為預設不選標籤）
        enable_tgant: false,    // 預設關閉 TGANT（因為預設不選標籤）
        generate_multiplier: '100',  // 預設 100x
        tgant_threshold: 0.95,  // TGANT 收斂門檻
        tgant_max_gen: 10,      // TGANT 最大迭代代數
        tgant_metric: 'min',    // TGANT 評估指標 (min/macro)
        use_auto_params: true,  // 預設使用自動參數
        ctgan: {
          // NeurIPS 2019 論文最佳實踐 + SDV 官方預設
          // 自動參數會根據樣本數動態調整
          epochs: '500',          // 基礎值提高，自動調整會倍增 (n<50: ×15, n<100: ×10, ...)
          batch_size: '500',      // 必須是 10 的倍數，自動調整會根據樣本數縮小
          discriminator_steps: '1', // 自動調整: n<100 用 3, n<500 用 2
          generator_dim: '256',   // 自動調整: n<50 用 32, n<100 用 64, n<300 用 128
          discriminator_dim: '256',
          embedding_dim: '128',
          generator_lr: '0.0002', // SDV 預設
          discriminator_lr: '0.0002'
        },
        tvae: {
          // NeurIPS 2019 論文最佳實踐 + SDV 官方預設
          // VAE 訓練收斂較慢，需要更多 epochs
          epochs: '500',          // 基礎值提高，自動調整會倍增 (n<50: ×20, n<100: ×15, ...)
          batch_size: '500',      // 必須是 10 的倍數
          compress_dims: '128',   // 自動調整: n<50 用 32, n<100 用 64
          decompress_dims: '128',
          embedding_dim: '128',
          l2scale: '0.00001',     // 1e-5 (正則化強度)
          loss_factor: '2'        // 重建損失權重
        },
        tabddpm: {
          // ICML 2023 論文 Table 1 最佳實踐
          // 論文建議: n_iter={5000,10000,20000}, batch={256,4096}, timesteps={100,1000}
          // 預設 5000: 論文建議下限，平衡訓練時間與品質
          n_iter: '5000',         // 論文建議下限，可手動調整至 10000/20000 獲得更高品質
          batch_size: '256',      // 論文建議值
          num_timesteps: '100'    // 100=快速模式, 1000=高品質模式
        }
      },
      preview_data: {
        columns: [],
        preview: [],
        total_rows: 0,
        total_columns: 0
      },
      summary: {},
      errors: {},
      output: null,
      modal: {
        title: '',
        content: '',
        icon: 'success'
      }
    }
  },
  computed: {
    // 資料規模等級判斷 - 分開評估欄位數和樣本數
    dataSizeLevel() {
      const rows = this.preview_data.total_rows || 0
      const cols = this.preview_data.total_columns || 0
      
      if (rows === 0 || cols === 0) {
        return { label: '-', class: 'bg-secondary', tip: '' }
      }
      
      // 欄位數是影響訓練時間的主要因素（尤其對 TabDDPM）
      // 樣本數影響較小，但也要考慮
      let colLevel = 0  // 0=小, 1=中, 2=大, 3=超大, 4=巨型
      let rowLevel = 0
      
      // 欄位數分級（對訓練時間影響最大）
      if (cols <= 20) colLevel = 0
      else if (cols <= 50) colLevel = 1
      else if (cols <= 100) colLevel = 2
      else if (cols <= 200) colLevel = 3
      else colLevel = 4
      
      // 樣本數分級
      if (rows <= 5000) rowLevel = 0
      else if (rows <= 20000) rowLevel = 1
      else if (rows <= 100000) rowLevel = 2
      else if (rows <= 500000) rowLevel = 3
      else rowLevel = 4
      
      // 綜合評估（欄位數權重較高）
      const level = Math.max(colLevel, Math.floor((colLevel * 2 + rowLevel) / 3))
      
      const levels = [
        { label: this.$t('lblSizeSmall') || '小型', class: 'bg-success', tip: this.$t('lblSizeSmallTip') || '訓練較快' },
        { label: this.$t('lblSizeMedium') || '中型', class: 'bg-info', tip: this.$t('lblSizeMediumTip') || '訓練時間適中' },
        { label: this.$t('lblSizeLarge') || '大型', class: 'bg-warning text-dark', tip: this.$t('lblSizeLargeTip') || '訓練較慢' },
        { label: this.$t('lblSizeHuge') || '超大型', class: 'bg-danger', tip: this.$t('lblSizeHugeTip') || '訓練很慢' },
        { label: this.$t('lblSizeMassive') || '巨型', class: 'bg-dark', tip: this.$t('lblSizeMassiveTip') || '訓練極慢，建議分批處理' }
      ]
      
      return levels[Math.min(level, 4)]
    },
    // 欄位數警告等級
    columnWarningLevel() {
      const cols = this.preview_data.total_columns || 0
      if (cols <= 30) return null
      if (cols <= 50) return { level: 'info', msg: this.$t('msgColsModerate') || '欄位數適中' }
      if (cols <= 100) return { level: 'warning', msg: this.$t('msgColsHigh') || '欄位較多，TabDDPM 訓練會較慢' }
      return { level: 'danger', msg: this.$t('msgColsVeryHigh') || '欄位數很多，強烈建議使用 CTGAN/TVAE 或考慮特徵篩選' }
    },
    // 樣本數警告等級  
    rowWarningLevel() {
      const rows = this.preview_data.total_rows || 0
      if (rows <= 10000) return null
      if (rows <= 50000) return { level: 'info', msg: this.$t('msgRowsModerate') || '樣本數適中' }
      if (rows <= 200000) return { level: 'warning', msg: this.$t('msgRowsHigh') || '樣本數較多，訓練時間會增加' }
      return { level: 'danger', msg: this.$t('msgRowsVeryHigh') || '樣本數很多，建議考慮抽樣或分批處理' }
    },
    // 訓練開始時間格式化（保留但簡化）
    generatorDescription() {
      const descriptions = {
        ctgan: this.$t('msgCtganDescription') || 'CTGAN 使用條件生成對抗網路生成表格數據，適合處理類別不平衡問題（NeurIPS 2019）',
        tvae: this.$t('msgTvaeDescription') || 'TVAE 使用變分自編碼器生成表格數據，訓練穩定適合小數據集（NeurIPS 2019）',
        tabddpm: this.$t('msgTabddpmDescription') || 'TabDDPM 使用擴散模型生成表格數據，品質最高但訓練時間較長（ICML 2023）'
      }
      return descriptions[this.selected.generator_type] || ''
    },
    syntheticPreviewData() {
      // 將後端的 records 格式轉換為 columns + rows 格式
      if (!this.output || !this.output.synthetic_preview || !Array.isArray(this.output.synthetic_preview)) {
        return { columns: [], rows: [] }
      }
      const rows = this.output.synthetic_preview
      const columns = rows.length > 0 ? Object.keys(rows[0]) : []
      return { columns, rows }
    },
    // 條件規則：根據 when 欄位類型返回可用的運算子
    availableWhenOperators() {
      const col = this.conditionalDraft.when.column
      if (!col) return ['==']
      if (this.isNumericColumn(col)) {
        return ['==', '!=', '>=', '<=', '>', '<', 'between']
      }
      return ['==', '!=', 'in', 'not_in']
    },
    whenNeedsSingleValue() {
      const op = this.conditionalDraft.when.operator
      return ['==', '!=', '>=', '<=', '>', '<'].includes(op)
    },
    whenNeedsValuesList() {
      const op = this.conditionalDraft.when.operator
      return ['in', 'not_in'].includes(op)
    },
    // TabDDPM 即時偏離警告（手動模式用）
    tabddpmDeviationWarnings() {
      const warnings = []
      const t = this.selected.tabddpm
      const nIter = parseInt(t.n_iter, 10)
      const batch = parseInt(t.batch_size, 10)
      const timesteps = parseInt(t.num_timesteps, 10)

      if (!isNaN(nIter) && nIter < 5000) {
        warnings.push(
          this.$t('msgDdpmWarnNiter', { value: nIter }) ||
          `n_iter=${nIter} 低於論文建議下限 5000（論文 Table 1: 5000/10000/20000）`
        )
      }
      if (!isNaN(batch) && batch !== 256 && batch !== 4096) {
        warnings.push(
          this.$t('msgDdpmWarnBatch', { value: batch }) ||
          `batch_size=${batch} 不是論文建議值（論文 Table 1: 256 或 4096）`
        )
      }
      if (!isNaN(timesteps) && timesteps !== 100 && timesteps !== 1000) {
        warnings.push(
          this.$t('msgDdpmWarnTimesteps', { value: timesteps }) ||
          `num_timesteps=${timesteps} 不是論文建議值（論文 Table 1: 100 或 1000）`
        )
      }
      return warnings
    },
    // 數值約束即時警告：當使用者輸入的值超出原始資料範圍時顯示
    // 生成模型無法外插，超出範圍會導致 post-filter clip 產生邊界值堆疊（垃圾資料）
    constraintNumericWarning() {
      const col = this.constraintSelectedColumn
      if (!col || !this.isNumericColumn(col)) return ''
      const stats = this.summary[col]
      if (!stats || stats.min === null || stats.max === null) return ''

      const origMin = Number(stats.min)
      const origMax = Number(stats.max)
      const d = this.constraintDraft

      // 檢查各種 type 的超出情況
      const issues = []
      const checkVal = (v, label) => {
        if (v === '' || v === null || v === undefined) return
        const n = Number(v)
        if (isNaN(n)) return
        if (n < origMin || n > origMax) {
          issues.push(`${label}=${n}`)
        }
      }

      if (d.type === 'range') {
        checkVal(d.min, this.$t('lblMin') || 'min')
        checkVal(d.max, this.$t('lblMax') || 'max')
      } else if (d.type === 'min' || d.type === 'max' || d.type === 'fixed') {
        checkVal(d.value, d.type)
      }

      if (issues.length === 0) return ''
      return (
        this.$t('msgConstraintOutOfRange', {
          col: col,
          values: issues.join(', '),
          origMin: origMin,
          origMax: origMax
        }) ||
        `⚠️ ${issues.join(', ')} 超出 ${col} 的原始範圍 [${origMin}, ${origMax}]。生成模型無法外插，post-filter clip 會將所有值硬壓到邊界，導致資料品質崩壞。建議調整到原始範圍內。`
      )
    }
  },
  watch: {
    "selected.data"() {
      this.selected.label_column = 'none'  // 清空標籤選擇
      if (this.selected.data != '') {
        this.previewTab()
      } else {
        this.initPreviewData()
      }
    },
    "selected.label_column"(newVal) {
      // 沒選 label 時：自動關 TGANT、切 mixed
      if (!newVal || newVal === 'none') {
        this.selected.enable_tgant = false
        this.selected.train_mode = 'mixed'
      } else {
        // 選了 label：預設開啟 TGANT、切 grouped
        this.selected.enable_tgant = true
        this.selected.train_mode = 'grouped'
      }
    },
    "selected.train_mode"(newVal) {
      // 切換到分群模式時，如果開啟自動參數，重置學習率
      if (newVal === 'grouped' && this.selected.use_auto_params) {
        this.selected.ctgan.generator_lr = '0.0002'
        this.selected.ctgan.discriminator_lr = '0.0002'
      }
    }
  },
  mounted() {
    this.fetchFiles()
    // 添加頁面離開警告
    window.addEventListener('beforeunload', this.handleBeforeUnload)
  },
  beforeUnmount() {
    // 移除頁面離開警告
    window.removeEventListener('beforeunload', this.handleBeforeUnload)
  },
  methods: {
    // 頁面離開警告處理
    handleBeforeUnload(e) {
      if (this.loading) {
        e.preventDefault()
        e.returnValue = '訓練正在進行中，確定要離開嗎？'
        return e.returnValue
      }
    },
    
    // 格式化百分比，安全處理 null/undefined/NaN
    formatPercent(value) {
      if (value === null || value === undefined || isNaN(value)) {
        return '-'
      }
      return (value * 100).toFixed(2) + '%'
    },

    initPreviewData() {
      this.preview_data = {
        columns: [],
        preview: [],
        total_rows: 0,
        total_columns: 0
      }
      this.summary = {}
    },

    // 重置為預設參數
    resetToDefaults() {
      // 根據是否選擇標籤欄位決定訓練模式
      if (this.selected.label_column && this.selected.label_column !== 'none') {
        this.selected.train_mode = 'grouped'
        this.selected.enable_tgant = true
      } else {
        this.selected.train_mode = 'mixed'
        this.selected.enable_tgant = false
      }
      // CTGAN 預設值 (NeurIPS 2019 + SDV 官方)
      this.selected.ctgan.epochs = '500'
      this.selected.ctgan.generator_lr = '0.0002'
      this.selected.ctgan.discriminator_lr = '0.0002'
      this.selected.ctgan.generator_dim = '256'
      this.selected.ctgan.discriminator_dim = '256'
      this.selected.ctgan.embedding_dim = '128'
      this.selected.ctgan.batch_size = '500'
      this.selected.ctgan.discriminator_steps = '1'
      // TVAE 預設值 (NeurIPS 2019 + SDV 官方)
      this.selected.tvae.epochs = '500'
      this.selected.tvae.batch_size = '500'
      this.selected.tvae.compress_dims = '128'
      this.selected.tvae.decompress_dims = '128'
      this.selected.tvae.embedding_dim = '128'
      this.selected.tvae.l2scale = '0.00001'
      this.selected.tvae.loss_factor = '2'
      // TabDDPM 預設值 (ICML 2023 Table 1)
      this.selected.tabddpm.n_iter = '5000'  // 論文建議下限
      this.selected.tabddpm.batch_size = '256'
      this.selected.tabddpm.num_timesteps = '100'
      // TGANT 預設值
      this.selected.generate_multiplier = '100'
      this.selected.tgant_threshold = 0.95
      this.selected.tgant_max_gen = 10
      this.selected.tgant_metric = 'min'
      this.selected.use_auto_params = true
      this.showAdvanced = true
    },

    toggleCollapse() {
      let collapseElement = this.$refs.collapsePreview
      let collapseInstance = Collapse.getInstance(collapseElement) || new Collapse(collapseElement)
      collapseInstance.toggle()
    },

    async fetchFiles() {
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/list-files`, {
          folder_path: `upload/${sessionStorage.getItem('username')}`,
          ext1: 'csv',
          ext2: 'xlsx'
        })
        if (response.data.status === 'success') {
          this.files = response.data.files
        }
      } catch (error) {
        console.error('Error fetching files:', error)
      }
    },

    async previewTab() {
      this.initPreviewData()
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/preview-tabular`, {
          file_path: `upload/${sessionStorage.getItem('username')}/${this.selected.data}`
        })
        if (response.data.status === 'success') {
          this.preview_data = response.data.preview_data
          this.summary = response.data.summary || {}
        } else if (response.data.status === 'error') {
          this.modal.title = this.$t('lblError')
          // 顯示完整錯誤訊息
          let errorContent = response.data.message || 'Unknown error'
          if (response.data.stderr && response.data.stderr !== 'empty') {
            errorContent += '\n\n[stderr]\n' + response.data.stderr
          }
          if (response.data.stdout && response.data.stdout !== 'empty') {
            errorContent += '\n\n[stdout]\n' + response.data.stdout
          }
          this.modal.content = errorContent
          this.modal.icon = 'error'
          this.initPreviewData()
          this.selected.data = ''
          this.openModalNotification()
        }
      } catch (error) {
        this.modal.title = this.$t('lblError')
        this.modal.content = String(error)
        this.modal.icon = 'error'
        this.openModalNotification()
        this.initPreviewData()
        this.selected.data = ''
      }
      this.loading = false
    },

    isInt(value) {
      return /^[1-9][0-9]*$/.test(value)
    },

    // ============================================================
    // 欄位約束相關方法
    // ============================================================
    isNumericColumn(col) {
      // 用 summary.mean 判斷：數值欄位才有有效的 mean，類別欄位為 null
      return this.summary[col] && this.summary[col].mean !== null && this.summary[col].mean !== undefined
    },

    isColumnConstrained(col) {
      return this.constraintRules.some(r => r.column === col)
    },

    getColumnUniqueValues(col) {
      // 從 preview_data.preview 抽取唯一值（適用類別欄位）
      // 注意：只回傳原始資料中實際存在的值。生成模型無法產生原本不存在的類別，
      // 因此 UI 上也不提供手動新增新類別的選項。
      if (!this.preview_data.preview || this.preview_data.preview.length === 0) return []
      const values = new Set()
      for (const row of this.preview_data.preview) {
        if (row[col] !== null && row[col] !== undefined && row[col] !== '') {
          values.add(String(row[col]))
        }
      }
      return Array.from(values).sort()
    },

    onConstraintColumnChange() {
      // 切換欄位時重置 draft，並根據類型設定預設 type
      if (!this.constraintSelectedColumn) return
      this.constraintDraft = {
        type: this.isNumericColumn(this.constraintSelectedColumn) ? 'range' : 'options',
        min: '',
        max: '',
        value: '',
        values: []
      }
      this.errors.constraint = ''
    },

    resetConstraintDraft() {
      this.constraintSelectedColumn = ''
      this.constraintDraft = {
        type: 'range',
        min: '',
        max: '',
        value: '',
        values: []
      }
      this.errors.constraint = ''
    },

    addConstraintRule() {
      const col = this.constraintSelectedColumn
      const d = this.constraintDraft
      this.errors.constraint = ''

      if (!col) {
        this.errors.constraint = this.$t('msgValRequired')
        return
      }

      // 驗證並建立規則物件
      let rule = { column: col, type: d.type }

      if (d.type === 'range') {
        const hasMin = d.min !== '' && d.min !== null
        const hasMax = d.max !== '' && d.max !== null
        if (!hasMin && !hasMax) {
          this.errors.constraint = this.$t('msgConstraintRangeRequired')
          return
        }
        if (hasMin) rule.min = Number(d.min)
        if (hasMax) rule.max = Number(d.max)
        if (hasMin && hasMax && rule.min > rule.max) {
          this.errors.constraint = this.$t('msgConstraintMinMaxInvalid')
          return
        }
      } else if (d.type === 'min' || d.type === 'max') {
        if (d.value === '' || d.value === null) {
          this.errors.constraint = this.$t('msgValRequired')
          return
        }
        rule.value = Number(d.value)
      } else if (d.type === 'fixed') {
        if (d.value === '' || d.value === null) {
          this.errors.constraint = this.$t('msgValRequired')
          return
        }
        rule.value = this.isNumericColumn(col) ? Number(d.value) : d.value
      } else if (d.type === 'options') {
        if (!d.values || d.values.length === 0) {
          this.errors.constraint = this.$t('msgConstraintOptionsRequired')
          return
        }
        rule.values = [...d.values]
      }

      this.constraintRules.push(rule)
      this.resetConstraintDraft()
    },

    removeConstraintRule(index) {
      this.constraintRules.splice(index, 1)
    },

    getConstraintTypeLabel(type) {
      const map = {
        range: this.$t('lblConstraintRange'),
        min: this.$t('lblConstraintMin'),
        max: this.$t('lblConstraintMax'),
        fixed: this.$t('lblConstraintFixed'),
        options: this.$t('lblConstraintOptions')
      }
      return map[type] || type
    },

    formatConstraintValue(rule) {
      if (rule.type === 'range') {
        const lo = rule.min !== undefined ? rule.min : '-∞'
        const hi = rule.max !== undefined ? rule.max : '+∞'
        return `${lo} ~ ${hi}`
      }
      if (rule.type === 'min') return `≥ ${rule.value}`
      if (rule.type === 'max') return `≤ ${rule.value}`
      if (rule.type === 'fixed') return `= ${rule.value}`
      if (rule.type === 'options') return rule.values.join(', ')
      return ''
    },

    buildColumnConstraintsPayload() {
      // 轉換 constraintRules + conditionalRules 為後端期望的新格式
      const hasColumns = this.constraintRules && this.constraintRules.length > 0
      const hasConditionals = this.conditionalRules && this.conditionalRules.length > 0
      if (!hasColumns && !hasConditionals) return null

      // 單欄位約束
      const columns = {}
      if (hasColumns) {
        for (const rule of this.constraintRules) {
          const obj = { type: rule.type }
          if (rule.type === 'range') {
            if (rule.min !== undefined) obj.min = rule.min
            if (rule.max !== undefined) obj.max = rule.max
          } else if (rule.type === 'min' || rule.type === 'max' || rule.type === 'fixed') {
            obj.value = rule.value
          } else if (rule.type === 'options') {
            obj.values = rule.values
          }
          columns[rule.column] = obj
        }
      }

      // 條件規則
      const conditionals = []
      if (hasConditionals) {
        for (const rule of this.conditionalRules) {
          conditionals.push({
            when: { ...rule.when },
            then: { ...rule.then }
          })
        }
      }

      return { columns, conditionals }
    },

    // ============================================================
    // 條件規則相關方法
    // ============================================================
    openConditionalForm() {
      this.showConditionalForm = true
      this.resetConditionalDraftValues()
    },

    resetConditionalDraftValues() {
      this.conditionalDraft = {
        when: { column: '', operator: '==', value: '', values: [], min: '', max: '' },
        then: { column: '', type: 'range', min: '', max: '', value: '', values: [] }
      }
      this.errors.conditional = ''
    },

    resetConditionalDraft() {
      // 取消按鈕：關閉表單並重置
      this.showConditionalForm = false
      this.resetConditionalDraftValues()
    },

    onWhenColumnChange() {
      // 切換 when 欄位時重置運算子和值
      const col = this.conditionalDraft.when.column
      if (!col) return
      // 設定預設運算子（依欄位類型）
      this.conditionalDraft.when.operator = this.isNumericColumn(col) ? '>=' : '=='
      this.conditionalDraft.when.value = ''
      this.conditionalDraft.when.values = []
      this.conditionalDraft.when.min = ''
      this.conditionalDraft.when.max = ''
      this.errors.conditional = ''
    },

    onThenColumnChange() {
      // 切換 then 欄位時重置類型和值
      const col = this.conditionalDraft.then.column
      if (!col) return
      this.conditionalDraft.then.type = this.isNumericColumn(col) ? 'range' : 'options'
      this.conditionalDraft.then.min = ''
      this.conditionalDraft.then.max = ''
      this.conditionalDraft.then.value = ''
      this.conditionalDraft.then.values = []
      this.errors.conditional = ''
    },

    addConditionalRule() {
      const d = this.conditionalDraft
      this.errors.conditional = ''

      // 驗證 when
      if (!d.when.column) {
        this.errors.conditional = this.$t('msgConditionalIncomplete')
        return
      }
      if (!d.when.operator) {
        this.errors.conditional = this.$t('msgConditionalIncomplete')
        return
      }

      const op = d.when.operator
      const whenObj = { column: d.when.column, operator: op }

      if (['==', '!=', '>=', '<=', '>', '<'].includes(op)) {
        if (d.when.value === '' || d.when.value === null || d.when.value === undefined) {
          this.errors.conditional = this.$t('msgConditionalIncomplete')
          return
        }
        whenObj.value = this.isNumericColumn(d.when.column) ? Number(d.when.value) : d.when.value
      } else if (['in', 'not_in'].includes(op)) {
        if (!d.when.values || d.when.values.length === 0) {
          this.errors.conditional = this.$t('msgWhenValuesRequired')
          return
        }
        whenObj.values = [...d.when.values]
      } else if (op === 'between') {
        if (d.when.min === '' || d.when.max === '') {
          this.errors.conditional = this.$t('msgConditionalIncomplete')
          return
        }
        const lo = Number(d.when.min)
        const hi = Number(d.when.max)
        if (lo > hi) {
          this.errors.conditional = this.$t('msgConstraintMinMaxInvalid')
          return
        }
        whenObj.min = lo
        whenObj.max = hi
      }

      // 驗證 then
      if (!d.then.column) {
        this.errors.conditional = this.$t('msgConditionalIncomplete')
        return
      }
      if (!d.then.type) {
        this.errors.conditional = this.$t('msgConditionalIncomplete')
        return
      }

      const thenObj = { column: d.then.column, type: d.then.type }

      if (d.then.type === 'range') {
        const hasMin = d.then.min !== '' && d.then.min !== null
        const hasMax = d.then.max !== '' && d.then.max !== null
        if (!hasMin && !hasMax) {
          this.errors.conditional = this.$t('msgConstraintRangeRequired')
          return
        }
        if (hasMin) thenObj.min = Number(d.then.min)
        if (hasMax) thenObj.max = Number(d.then.max)
        if (hasMin && hasMax && thenObj.min > thenObj.max) {
          this.errors.conditional = this.$t('msgConstraintMinMaxInvalid')
          return
        }
      } else if (d.then.type === 'min' || d.then.type === 'max') {
        if (d.then.value === '' || d.then.value === null) {
          this.errors.conditional = this.$t('msgConditionalIncomplete')
          return
        }
        thenObj.value = Number(d.then.value)
      } else if (d.then.type === 'fixed') {
        if (d.then.value === '' || d.then.value === null) {
          this.errors.conditional = this.$t('msgConditionalIncomplete')
          return
        }
        thenObj.value = this.isNumericColumn(d.then.column) ? Number(d.then.value) : d.then.value
      } else if (d.then.type === 'options') {
        if (!d.then.values || d.then.values.length === 0) {
          this.errors.conditional = this.$t('msgConstraintOptionsRequired')
          return
        }
        thenObj.values = [...d.then.values]
      }

      this.conditionalRules.push({ when: whenObj, then: thenObj })
      this.showConditionalForm = false
      this.resetConditionalDraftValues()
    },

    removeConditionalRule(index) {
      this.conditionalRules.splice(index, 1)
    },

    getOperatorLabel(op) {
      const map = {
        '==': this.$t('lblOpEq'),
        '!=': this.$t('lblOpNeq'),
        '>=': this.$t('lblOpGte'),
        '<=': this.$t('lblOpLte'),
        '>': this.$t('lblOpGt'),
        '<': this.$t('lblOpLt'),
        'in': this.$t('lblOpIn'),
        'not_in': this.$t('lblOpNotIn'),
        'between': this.$t('lblOpBetween')
      }
      return map[op] || op
    },

    formatWhenClause(when) {
      const op = when.operator
      if (['==', '!=', '>=', '<=', '>', '<'].includes(op)) {
        return `${when.column} ${op} ${when.value}`
      }
      if (op === 'in') return `${when.column} ∈ {${when.values.join(', ')}}`
      if (op === 'not_in') return `${when.column} ∉ {${when.values.join(', ')}}`
      if (op === 'between') return `${when.column} ∈ [${when.min}, ${when.max}]`
      return ''
    },

    formatThenClause(then) {
      const col = then.column
      const t = then.type
      if (t === 'range') {
        const lo = then.min !== undefined ? then.min : '-∞'
        const hi = then.max !== undefined ? then.max : '+∞'
        return `${col} ∈ [${lo}, ${hi}]`
      }
      if (t === 'min') return `${col} ≥ ${then.value}`
      if (t === 'max') return `${col} ≤ ${then.value}`
      if (t === 'fixed') return `${col} = ${then.value}`
      if (t === 'options') return `${col} ∈ {${then.values.join(', ')}}`
      return ''
    },

    validateForm() {
      this.errors = {}
      let isValid = true

      if (!this.selected.generator_type) {
        this.errors.generator_type = this.$t('msgValRequired')
        isValid = false
      }

      if (!this.selected.data) {
        this.errors.data = this.$t('msgValRequired')
        isValid = false
      }

      if (this.selected.num_samples_type === 'custom') {
        if (!this.selected.custom_num_samples || !this.isInt(this.selected.custom_num_samples)) {
          this.errors.custom_num_samples = this.$t('msgValIntOnly')
          isValid = false
        }
      }

      if (this.selected.generator_type === 'ctgan') {
        if (!this.selected.ctgan.epochs || !this.isInt(this.selected.ctgan.epochs)) {
          this.errors.epochs = this.$t('msgValIntOnly')
          isValid = false
        }
        if (!this.selected.ctgan.batch_size || !this.isInt(this.selected.ctgan.batch_size)) {
          this.errors.batch_size = this.$t('msgValIntOnly')
          isValid = false
        }
      }

      if (this.selected.generator_type === 'tvae') {
        if (!this.selected.tvae.epochs || !this.isInt(this.selected.tvae.epochs)) {
          this.errors.tvae_epochs = this.$t('msgValIntOnly')
          isValid = false
        }
        if (!this.selected.tvae.batch_size || !this.isInt(this.selected.tvae.batch_size)) {
          this.errors.tvae_batch_size = this.$t('msgValIntOnly')
          isValid = false
        }
      }

      return isValid
    },

    async runGenerate() {
      if (!this.validateForm()) {
        return
      }

      this.loading = true
      this.output = null

      try {
        // 根據生成器類型選擇參數
        const genType = this.selected.generator_type
        const isCtgan = genType === 'ctgan'
        const isTvae = genType === 'tvae'
        const isTabddpm = genType === 'tabddpm'
        
        // 選擇對應的參數組
        let params
        if (isCtgan) {
          params = this.selected.ctgan
        } else if (isTvae) {
          params = this.selected.tvae
        } else if (isTabddpm) {
          params = this.selected.tabddpm
        } else {
          params = this.selected.ctgan  // 預設
        }
        
        // 構建請求參數
        const requestData = {
          file_path: `upload/${sessionStorage.getItem('username')}/${this.selected.data}`,
          label_column: this.selected.label_column,  // 單標籤
          generator_type: genType,
          num_samples: this.selected.num_samples_type === 'custom' 
            ? this.selected.custom_num_samples 
            : this.selected.num_samples_type,
          // 進階參數
          train_mode: this.selected.train_mode,
          enable_tgant: this.selected.enable_tgant,
          generate_multiplier: this.selected.generate_multiplier,
          tgant_threshold: this.selected.tgant_threshold,
          tgant_max_gen: this.selected.tgant_max_gen,
          tgant_metric: this.selected.tgant_metric,
          use_auto_params: this.selected.use_auto_params,
          username: sessionStorage.getItem('username')
        }

        // 欄位約束（如果有設定）
        const columnConstraintsPayload = this.buildColumnConstraintsPayload()
        if (columnConstraintsPayload) {
          requestData.column_constraints = columnConstraintsPayload
        }
        
        // 根據生成器類型添加特定參數
        if (isTabddpm) {
          // TabDDPM 參數映射：
          // - n_iter -> epochs (後端會再映射)
          // - num_timesteps -> generator_dim (後端會判斷並使用)
          requestData.epochs = params.n_iter
          requestData.batch_size = params.batch_size
          requestData.generator_dim = params.num_timesteps  // timesteps 通過 generator_dim 傳遞
          // 其他參數設為預設值（後端會忽略）
          requestData.embedding_dim = '128'
          requestData.discriminator_steps = '1'
          requestData.discriminator_dim = '128'
          requestData.generator_lr = '0.0002'
          requestData.discriminator_lr = '0.0002'
        } else {
          // CTGAN / TVAE 參數
          requestData.epochs = params.epochs
          requestData.batch_size = params.batch_size
          requestData.embedding_dim = params.embedding_dim
          requestData.discriminator_steps = isCtgan ? params.discriminator_steps : '1'
          requestData.generator_dim = isCtgan ? params.generator_dim : params.compress_dims
          requestData.discriminator_dim = isCtgan ? params.discriminator_dim : params.decompress_dims
          requestData.generator_lr = isCtgan ? params.generator_lr : '0.0002'
          requestData.discriminator_lr = isCtgan ? params.discriminator_lr : '0.0002'
        }
        
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/run-synthetic`, requestData)

        if (response.data.status === 'success') {
          this.output = response.data
          this.modal.title = this.$t('lblSuccess')
          this.modal.content = this.$t('msgSyntheticSuccess')
          this.modal.icon = 'success'
          this.openModalNotification()
        } else {
          this.modal.title = this.$t('lblError')
          // 顯示完整錯誤訊息
          let errorContent = response.data.message || this.$t('msgSyntheticFailed')
          if (response.data.stderr && response.data.stderr !== 'empty') {
            errorContent += '\n\n[stderr]\n' + response.data.stderr
          }
          if (response.data.stdout && response.data.stdout !== 'empty') {
            errorContent += '\n\n[stdout]\n' + response.data.stdout
          }
          if (response.data.returncode !== undefined) {
            errorContent += '\n\n[returncode: ' + response.data.returncode + ']'
          }
          this.modal.content = errorContent
          this.modal.icon = 'error'
          this.openModalNotification()
        }
      } catch (error) {
        this.modal.title = this.$t('lblError')
        this.modal.content = error.response?.data?.message || String(error)
        this.modal.icon = 'error'
        this.openModalNotification()
      }

      this.loading = false
    },

    downloadUrl(filename) {
      if (!this.output) return '#'
      return `${process.env.VUE_APP_API_URL}/download-synthetic/${sessionStorage.getItem('username')}/${this.output.task_id}/${filename}`
    },

    getScoreClass(score) {
      if (score == null) return ''
      if (score >= 0.9) return 'text-success fw-bold'
      if (score >= 0.7) return 'text-primary'
      if (score >= 0.5) return 'text-warning'
      return 'text-danger'
    },

    // Discriminator Measure 顏色評分 (越接近 50% 越好)
    getDiscriminatorClass(accuracy) {
      if (accuracy == null) return ''
      const diff = Math.abs(accuracy - 0.5)  // 與 50% 的差距
      if (diff <= 0.05) return 'text-success fw-bold'  // 45-55%
      if (diff <= 0.15) return 'text-primary'           // 35-45% 或 55-65%
      if (diff <= 0.25) return 'text-warning'           // 25-35% 或 65-75%
      return 'text-danger'                              // <25% 或 >75%
    },
    
    getLastRecall(history, type) {
      // 兼容新舊格式：recall_trts_macro / recall_trts
      if (!history || history.length === 0) return '-'
      const last = history[history.length - 1]
      const key = type === 'trts' 
        ? (last.recall_trts_macro !== undefined ? 'recall_trts_macro' : 'recall_trts')
        : (last.recall_tstr_macro !== undefined ? 'recall_tstr_macro' : 'recall_tstr')
      const val = last[key]
      return val != null ? (val * 100).toFixed(1) : '-'
    },

    formatTrainingTime(seconds) {
      if (seconds == null) return '-'
      if (seconds < 60) {
        return `${seconds.toFixed(1)} sec`
      } else if (seconds < 3600) {
        const mins = Math.floor(seconds / 60)
        const secs = Math.round(seconds % 60)
        return `${mins} min ${secs} sec`
      } else {
        const hours = Math.floor(seconds / 3600)
        const mins = Math.round((seconds % 3600) / 60)
        return `${hours} hr ${mins} min`
      }
    },

    getDiscriminatorAccuracy(model) {
      try {
        const dm = this.output?.evaluation?.utility?.discriminator_measure
        if (!dm) return '-'
        const acc = dm[model]?.accuracy
        if (acc == null) return '-'
        return (acc * 100).toFixed(2)
      } catch {
        return '-'
      }
    },

    openModalNotification() {
      this.$refs.modalNotificationRef.openModal()
    }
  }
}
</script>

<style scoped>
.results-section {
  margin-top: 20px;
}

.card {
  margin-bottom: 1rem;
}

.sticky-top {
  position: sticky;
  top: 0;
  z-index: 1;
}

.table-responsive {
  border: 1px solid #dee2e6;
  border-radius: 0.25rem;
}
</style>