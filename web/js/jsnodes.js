import { app } from "../../../scripts/app.js";

// 工具类：错误处理和性能优化
class NodeUtils {
    /**
     * 安全执行函数并处理错误
     * @param {Function} fn - 要执行的函数
     * @param {string} context - 错误上下文
     * @returns {any} 执行结果
     */
    static safeExecute(fn, context = "") {
        try {
            return fn();
        } catch (error) {
            console.error(`Error in ${context}:`, error);
            NodeUtils.showUserError(`Error in ${context}: ${error.message}`);
            return null;
        }
    }
    
    /**
     * 显示用户友好的错误消息
     * @param {string} message - 错误消息
     */
    static showUserError(message) {
        if (typeof app !== 'undefined' && app.ui && app.ui.dialog) {
            app.ui.dialog.show("Error", message);
        } else {
            alert(message);
        }
    }
    
    /**
     * 验证数值输入
     * @param {number} value - 要验证的值
     * @param {number} min - 最小值
     * @param {number} max - 最大值
     * @param {string} name - 参数名称
     * @returns {boolean} 是否有效
     */
    static validateNumber(value, min, max, name) {
        if (typeof value !== 'number' || isNaN(value)) {
            NodeUtils.showUserError(`${name} must be a valid number`);
            return false;
        }
        if (value < min || value > max) {
            NodeUtils.showUserError(`${name} must be between ${min} and ${max}`);
            return false;
        }
        return true;
    }
    
    /**
     * 防抖函数
     * @param {Function} func - 要防抖的函数
     * @param {number} wait - 等待时间（毫秒）
     * @returns {Function} 防抖后的函数
     */
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    /**
     * 性能监控
     * @param {string} label - 性能标签
     * @returns {Function} 结束监控的函数
     */
    static performanceMonitor(label) {
        if (typeof performance !== 'undefined') {
            const startTime = performance.now();
            return () => {
                const endTime = performance.now();
                const duration = endTime - startTime;
                if (duration > 100) {
                    console.warn(`${label} took ${duration.toFixed(2)}ms`);
                }
            };
        }
        return () => {};
    }
}

// 节点管理器类
class NodeManager {
    constructor() {
        this.updateQueue = new Map();
        this.nodeCache = new WeakMap();
    }
    
    /**
     * 安全地更新节点
     * @param {Object} node - 节点对象
     * @param {Function} updateFn - 更新函数
     * @param {string} context - 更新上下文
     */
    safeUpdateNode(node, updateFn, context = "") {
        const nodeId = node.id || node.constructor.name;
        
        // 防止重复更新
        if (this.updateQueue.has(nodeId)) {
            return;
        }
        
        this.updateQueue.set(nodeId, true);
        
        NodeUtils.safeExecute(() => {
            const endMonitor = NodeUtils.performanceMonitor(`Node update ${nodeId}`);
            
            updateFn(node);
            
            // 更新节点UI
            if (typeof node.computeSize === 'function') {
                node.computeSize();
            }
            if (typeof node.setDirtyCanvas === 'function') {
                node.setDirtyCanvas(true, true);
            }
            
            endMonitor();
        }, context || `safeUpdateNode(${nodeId})`);
        
        // 清理更新队列
        setTimeout(() => {
            this.updateQueue.delete(nodeId);
        }, 100);
    }
    
    /**
     * 查找组件
     * @param {Object} node - 节点对象
     * @param {string} name - 组件名称
     * @returns {Object|null} 组件对象
     */
    findWidget(node, name) {
        return node.widgets?.find(w => w.name === name) || null;
    }
    
    /**
     * 查找输入
     * @param {Object} node - 节点对象
     * @param {string} name - 输入名称
     * @returns {Object|null} 输入对象
     */
    findInput(node, name) {
        return node.inputs?.find(input => input.name === name) || null;
    }
    
    /**
     * 移除动态输入和组件
     * @param {Object} node - 节点对象
     * @param {RegExp} namePattern - 名称模式
     */
    removeDynamicElements(node, namePattern) {
        // 移除输入
        const inputsToRemove = node.inputs?.filter(input => namePattern.test(input.name)) || [];
        inputsToRemove.forEach(input => {
            const index = node.inputs.indexOf(input);
            if (index > -1) {
                node.inputs.splice(index, 1);
            }
        });
        
        // 移除组件
        const widgetsToRemove = node.widgets?.filter(widget => namePattern.test(widget.name)) || [];
        widgetsToRemove.forEach(widget => {
            const index = node.widgets.indexOf(widget);
            if (index > -1) {
                node.widgets.splice(index, 1);
            }
        });
    }
}

// 创建全局节点管理器
const nodeManager = new NodeManager();

app.registerExtension({
    name: "loaders.jsnodes.ComfyCoupleMask",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 检查分类，支持英文和中文
        if (!nodeData?.category?.startsWith("loaders") && !nodeData?.category?.startsWith("加载器")) {
            return;
        }

        // ComfyCoupleMask节点处理
        if (nodeData.name === "ComfyCoupleMask") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // 添加"更新输入"按钮
                this.addWidget("button", "Update inputs", null, () => {
                    nodeManager.safeUpdateNode(this, (node) => {
                        // 获取期望的区域输入总数
                        const countWidget = nodeManager.findWidget(node, "inputcount");
                        if (!countWidget) {
                            console.error("ComfyCoupleMask: 找不到 'inputcount' 小部件！");
                            return;
                        }
                        
                        const desiredRegionCount = countWidget.value;
                        
                        // 验证输入数量
                        if (!NodeUtils.validateNumber(desiredRegionCount, 1, 20, "Input count")) {
                            countWidget.value = Math.max(1, Math.min(20, desiredRegionCount));
                            return;
                        }

                        // 计算当前已有的区域输入数量
                        const currentRegionInputs = node.inputs?.filter(input => input.name.startsWith("region_")) || [];
                        const currentRegionCount = currentRegionInputs.length;

                        // 如果数量已经匹配，则什么都不做
                        if (currentRegionCount === desiredRegionCount) {
                            return;
                        }

                        // 添加新的输入
                        if (desiredRegionCount > currentRegionCount) {
                            for (let i = currentRegionCount + 1; i <= desiredRegionCount; i++) {
                                node.addInput(`region_${i}`, "ATTENTION_COUPLE_REGION");
                            }
                        }
                        // 移除多余的输入
                        else {
                            // 从后往前移除，避免索引问题
                            for (let i = currentRegionCount; i > desiredRegionCount; i--) {
                                const lastIndex = node.inputs.length - 1;
                                if (lastIndex >= 0) {
                                    node.removeInput(lastIndex);
                                }
                            }
                        }
                    }, "ComfyCoupleMask update inputs");
                });
            };
        }

        // ComfyMultiRegion节点处理
        if (nodeData.name === "ComfyMultiRegion") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // 添加更新按钮
                this.addWidget("button", "Update Regions", null, () => {
                    updateRegions(this);
                });

                // 初始化区域
                const countWidget = nodeManager.findWidget(this, "num_regions");
                if (countWidget) {
                    // 设置默认值为2
                    if (countWidget.value === undefined) {
                        countWidget.value = 2;
                    }
                    // 初始化创建区域
                    updateRegions(this);
                }

                // 区域更新函数
                const updateRegions = NodeUtils.debounce((node) => {
                    nodeManager.safeUpdateNode(node, (node) => {
                        const countWidget = nodeManager.findWidget(node, "num_regions");
                        if (!countWidget) {
                            console.error("ComfyMultiRegion: 找不到 'num_regions' 小部件！");
                            return;
                        }
                        
                        const desiredRegionCount = countWidget.value;
                        
                        // 验证区域数量
                        if (!NodeUtils.validateNumber(desiredRegionCount, 1, 10, "Number of regions")) {
                            countWidget.value = Math.max(1, Math.min(10, desiredRegionCount));
                            return;
                        }

                        // 定义需要动态处理的输入和控件名称
                        const dynamicInputName = 'positive';
                        const dynamicWidgetNames = ['ratio', 'weight'];

                        // 计算当前已有的区域数量
                        let existingCount = 0;
                        while (node.inputs?.some(input => input.name === `${dynamicInputName}_${existingCount + 1}`)) {
                            existingCount++;
                        }

                        // 添加新区域
                        if (desiredRegionCount > existingCount) {
                            for (let i = existingCount + 1; i <= desiredRegionCount; i++) {
                                // 添加 positive 输入
                                node.addInput(`${dynamicInputName}_${i}`, 'CONDITIONING');

                                // 添加 ratio 控件
                                node.addWidget("number", `ratio_${i}`, 0.5, null, {
                                    min: 0.0,
                                    max: 1.0,
                                    step: 0.01,
                                    precision: 2
                                });

                                // 添加 weight 控件
                                node.addWidget("number", `weight_${i}`, 1.0, null, {
                                    min: 0.0,
                                    max: 10.0,
                                    step: 0.1,
                                    precision: 2
                                });
                            }
                        }
                        // 移除多余区域
                        else if (desiredRegionCount < existingCount) {
                            for (let i = existingCount; i > desiredRegionCount; i--) {
                                // 移除输入
                                const inputIndex = node.inputs?.findIndex(input => input.name === `${dynamicInputName}_${i}`);
                                if (inputIndex !== -1) {
                                    node.removeInput(inputIndex);
                                }

                                // 移除控件
                                dynamicWidgetNames.forEach(name => {
                                    const widgetIndex = node.widgets?.findIndex(w => w.name === `${name}_${i}`);
                                    if (widgetIndex !== -1) {
                                        node.widgets.splice(widgetIndex, 1);
                                    }
                                });
                            }
                        }
                        
                        // 验证比例总和
                        validateRatioSum(node);
                        
                    }, "ComfyMultiRegion update regions");
                }, 300);
                
                // 将更新函数附加到节点实例
                this.updateRegions = updateRegions;
            };
        }
    }
});

// 验证比例总和
function validateRatioSum(node) {
    const ratioWidgets = node.widgets?.filter(w => w.name.startsWith("ratio_")) || [];
    const sum = ratioWidgets.reduce((acc, widget) => acc + (widget.value || 0), 0);
    
    if (sum > 1.0) {
        NodeUtils.showUserError(`Warning: Ratio sum (${sum.toFixed(2)}) exceeds 1.0. Auto-adjusting ratios.`);
        
        // 自动调整比例
        ratioWidgets.forEach(widget => {
            widget.value = (widget.value || 0) / sum;
        });
    }
}

// 添加全局错误处理
window.addEventListener('error', function(event) {
    console.error('Global error caught:', event.error);
    NodeUtils.showUserError(`An unexpected error occurred: ${event.error.message}`);
});

// 添加未处理的Promise拒绝处理
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    NodeUtils.showUserError(`An unexpected error occurred: ${event.reason}`);
});

// 导出工具类和节点管理器（用于调试）
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NodeUtils, NodeManager };
}