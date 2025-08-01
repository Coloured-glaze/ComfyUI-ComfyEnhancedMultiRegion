import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "loaders.jsnodes.ComfyCoupleMask", // 使用一个更独特的名称
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 检查分类，支持英文和中文
        if (!nodeData?.category?.startsWith("loaders") && !nodeData?.category?.startsWith("加载器")) {
            return;
        }

        // ComfyCoupleMask节点处理
        else if (nodeData.name === "ComfyCoupleMask") {
            // 安全地“修补” onNodeCreated 函数，而不是直接覆盖它
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                // 首先调用原始的 onNodeCreated (如果存在)
                onNodeCreated?.apply(this, arguments);

                // 添加“更新输入”按钮
                this.addWidget("button", "Update inputs", null, () => {
                    // 1. 获取期望的区域输入总数
                    const countWidget = this.widgets.find(w => w.name === "inputcount");
                    if (!countWidget) {
                        console.error("ComfyCoupleMask: 找不到 'inputcount' 小部件！");
                        return;
                    }
                    const desiredRegionCount = countWidget.value;

                    // 2. 计算当前已有的区域输入数量
                    // 通过名字过滤，这是最精确和健壮的方法
                    const currentRegionInputs = this.inputs.filter(input => input.name.startsWith("region_"));
                    const currentRegionCount = currentRegionInputs.length;

                    // 3. 如果数量已经匹配，则什么都不做
                    if (currentRegionCount === desiredRegionCount) {
                        return;
                    }

                    // 4. 如果期望数量 > 当前数量，则添加新的输入
                    if (desiredRegionCount > currentRegionCount) {
                        for (let i = currentRegionCount + 1; i <= desiredRegionCount; i++) {
                            // 直接使用清晰的命名 `region_N`
                            this.addInput(`region_${i}`, "ATTENTION_COUPLE_REGION");
                        }
                    }
                    // 5. 如果期望数量 < 当前数量，则移除多余的输入
                    else {
                        // 从后往前移除，这是最安全的方式
                        for (let i = currentRegionCount; i > desiredRegionCount; i--) {
                            // 总是移除最后一个输入，避免索引问题
                            this.removeInput(this.inputs.length - 1);
                        }
                    }

                    // 触发UI更新
                    this.setDirtyCanvas(true, true);
                });
            };
        }

        // 添加ComfyMultiRegion节点处理
        if (nodeData.name === "ComfyMultiRegion") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // 添加更新按钮
                this.addWidget("button", "Update Regions", null, () => {
                    updateRegions(this);
                });

                // --- 新增：初始化区域 --- 
                const countWidget = this.widgets.find(w => w.name === "num_regions");
                if (countWidget) {
                    // 设置默认值为2
                    if (countWidget.value === undefined) {
                        countWidget.value = 2;
                    }
                    // 初始化创建区域
                    updateRegions(this);
                }

                // --- 提取动态区域更新逻辑为函数 --- 
                function updateRegions(node) {
                    const countWidget = node.widgets.find(w => w.name === "num_regions");
                    if (!countWidget) {
                        console.error("ComfyMultiRegion: 找不到 'num_regions' 小部件！");
                        return;
                    }
                    const desiredRegionCount = countWidget.value;

                    // 1. 定义需要动态处理的输入和控件名称
                    const dynamicInputName = 'positive';
                    const dynamicWidgetNames = ['ratio', 'weight'];

                    // 2. 计算当前已有的区域数量
                    let existingCount = 0;
                    while (true) {
                        const inputExists = node.inputs.some(input => input.name === `${dynamicInputName}_${existingCount + 1}`);
                        if (!inputExists) break;
                        existingCount++;
                    }

                    // 3. 根据需要添加新区域
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
                    // 4. 根据需要移除多余区域
                    else if (desiredRegionCount < existingCount) {
                        for (let i = existingCount; i > desiredRegionCount; i--) {
                            // 移除输入
                            const inputIndex = node.inputs.findIndex(input => input.name === `${dynamicInputName}_${i}`);
                            if (inputIndex !== -1) {
                                node.removeInput(inputIndex);
                            }

                            // 移除控件
                            dynamicWidgetNames.forEach(name => {
                                const widgetIndex = node.widgets.findIndex(w => w.name === `${name}_${i}`);
                                if (widgetIndex !== -1) {
                                    node.widgets.splice(widgetIndex, 1);
                                }
                            });
                        }
                    }

                    // 调整节点高度以适应新的控件
                    node.computeSize();
                    node.setDirtyCanvas(true, true);
                }
            };
        }
    }
});