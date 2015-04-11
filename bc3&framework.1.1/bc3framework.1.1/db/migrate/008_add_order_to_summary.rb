class AddOrderToSummary < ActiveRecord::Migration
  def self.up
    add_column :summaries, :list_order, :integer, :default => 1
  end

  def self.down
    remove_column :summaries, :list_order
  end
end
